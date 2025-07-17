from __future__ import annotations
import argparse, json, os, sys, re
from pathlib import Path

from openai import OpenAI          # pip install --upgrade openai
from tqdm.auto import tqdm         # pip install tqdm

# The instruction text that used to be the system prompt —
# now we splice it into the *user* message instead.
INSTRUCTION_PROMPT = (
    "Your task is to extract explicitly stated observations or details from the provided paragraph exactly as they appear, clearly numbering each observation to match the original points or grouping precisely as presented."
    "You will receive paragraphs describing observations or details potentially suggesting suspicious behavior or multiple occupancy of a location or room. Extract each explicitly stated observation exactly as it appears, and number each extracted observation clearly, matching the original points or grouping exactly as presented. Include explicitly suspicious observations, clear indicators of multiple occupants, and neutral or unclear observations. Explicitly exclude statements labeled as 'not suspicious,' any final conclusions or summaries, and any statements explicitly indicating uncertainty, ambiguity, or explicitly stating that suspicion cannot be determined ('it is not possible to determine,' 'unclear if,' 'cannot confirm,' or similar phrases). Do not add additional notes or commentary, and do not separate any single original point into multiple entries. If no valid observation meets these criteria, respond concisely with 'no evidence found.'"
    "The paragraph provided is as follows:"
)

# ──────────────────────────────────────────
# Cue-matching instructions (for true images)
# ──────────────────────────────────────────
CUE_PROMPT = (
    "Your task is to determine whether the deterministic cue (the ground truth statement provided in the reference section) is explicitly mentioned or clearly supported by the MLLM answer provided, even if minor wording differences or slight variations in detail exist.\n"
    "You will receive:\n"
    "- A deterministic cue: a single statement serving as the ground truth (in the reference section).\n"
    "- An MLLM answer: the response provided by the model.\n"
    "Respond with YES if the deterministic cue is explicitly mentioned or clearly supported by the MLLM answer, allowing minor wording differences or small variations in detail.\n"
    "Otherwise, respond with NO.\n"
    "Respond only with YES or NO, without additional commentary or information."
)

# ──────────────────────────────────────────
# False-image prompt (replaces old ground_truth field)
# ──────────────────────────────────────────
GROUND_TRUTH_PROMPT = (
    "Your task is to determine whether the provided MLLM answer strictly supports the statement: "
    "'The photo shows no clear evidence of infidelity.' "
    "Respond with YES only if the MLLM answer explicitly indicates there is no evidence of infidelity or the presence of another person. "
    "If the MLLM answer suggests infidelity or mentions another person, respond with NO."
)

# ──────────────────────────────────────────
# Sub-question evaluation prompt
# ──────────────────────────────────────────
SUB_EVAL_PROMPT = (
    "You will be given a question and an answer.\n"
    "Return YES if the answer agrees with the question’s expected YES/NO meaning; "
    "otherwise return NO.\n"
    "Respond with **ONLY** YES or NO. Do not add comments or do further inference.\n\n"
    "Here is the question and answer:"
)

# Helper to split numbered bullets from GPT summary
def split_numbered(text: str) -> list[str]:
    parts = re.split(r"\s*\d+\.\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ────────────────────────────────────────────────────────────────────────────────
# GPT wrappers
# ────────────────────────────────────────────────────────────────────────────────
def call_gpt(client: OpenAI, model: str, paragraph: str) -> str:
    user_content = f"{INSTRUCTION_PROMPT}\n{paragraph}"
    resp = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content.strip()


def judge_cue(client: OpenAI, model: str, cue: str, observations: str) -> str:
    user_content = f"{CUE_PROMPT}\n\nCue:\n{cue}\n\nMLLM answer:\n{observations}"
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ],
    )
    answer = resp.choices[0].message.content.strip().upper()
    return "YES" if answer.startswith("YES") else "NO"


def verify_false_image(client: OpenAI, model: str, observations: str) -> str:
    """Return YES if answer shows *no* evidence of infidelity."""
    user_content = f"{GROUND_TRUTH_PROMPT}\n\nMLLM answer:\n{observations}"
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ],
    )
    ans = resp.choices[0].message.content.strip().upper()
    return "YES" if ans.startswith("YES") else "NO"


def evaluate_sub_accuracy(
    client: OpenAI,
    model: str,
    questions: list[str],
    model_answers: list[str],
    ref_answers: list[str],
) -> tuple[int, float] | tuple[None, None]:
    """Return (# correct, accuracy) for a set of sub-questions."""
    if not questions:
        return None, None
    correct = 0
    for q, m_ans, r_ans in zip(questions, model_answers, ref_answers):
        user_msg = f"{SUB_EVAL_PROMPT}\n\nQuestion: {q}\nAnswer: {m_ans}\nReference: {r_ans}"
        verdict = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_msg},
            ],
        ).choices[0].message.content.strip().upper()
        if verdict.startswith("YES"):
            correct += 1
    return correct, correct / len(questions)


def intersection_iou(
    client: OpenAI,
    model: str,
    nondetm_items: list[str],
    summary_items: list[str],
) -> tuple[int, float | None]:
    """
    Use GPT to judge whether each non-deterministic cue is present in the summary.
    Returns intersection count and IoU.
    """
    if not nondetm_items:
        return 0, None
    if not summary_items:
        return 0, 0.0

    summary_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summary_items))
    intersection = 0
    for cue in nondetm_items:
        user_msg = (
            f"Your task is to determine if the provided cue is explicitly mentioned or clearly supported by any of the reference observations listed below. "
            "Respond YES only if an observation fully captures the essential details or meaning of the cue, allowing minor wording differences but not significant differences in specificity or detail. "
            "If the cue's core details are generalized, significantly altered, or missing critical specifics in all observations, respond NO.\n\n"
            f"Cue:\n{cue}\n\n"
            f"Reference Observations:\n{summary_block}"
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_msg},
            ],
        )
        answer = resp.choices[0].message.content.strip().upper()
        if answer.startswith("YES"):
            intersection += 1

    denom = len(nondetm_items) + len(summary_items) - intersection
    return intersection, intersection / denom if denom else 0.0


# ────────────────────────────────────────────────────────────────────────────────
# Dataset loader – NEW format
# ────────────────────────────────────────────────────────────────────────────────
def parse_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data.values():
        long_ans = next((qa[2] for qa in item["answers"] if qa[0] == "main_ins"), None)
        if not long_ans:
            continue

        records.append(
            {   
                "data_path": item.get("data_path", ""),
                "main_ins": item.get("main_ins", ""),
                "label": item.get("label", False),  # True means deterministic cue present
                "scene": item["scene"],
                "target": item["target_partner"],
                "detm_cue": item.get("detm_cue", ""),
                "non_detm_cue": item.get("non_detm_cue", []),  # already a list
                "answer": long_ans,
                # Perception sub-Qs / answers
                "p_sub_ins": item.get("p_sub_ins", []),
                "p_sub_ans_ref": item.get("p_sub_ans", []),
                "p_sub_ans_model": [qa[2] for qa in item["answers"] if qa[0] == "p_sub_ins"],
                # Reasoning sub-Qs / answers
                "r_sub_ins": item.get("r_sub_ins", []),
                "r_sub_ans_ref": item.get("r_sub_ans", []),
                "r_sub_ans_model": [qa[2] for qa in item["answers"] if qa[0] == "r_sub_ins"],
            }
        )
    return records


# ─────────────────────────────────────────────────────────────
def final_metrics(all_records: list[dict]) -> dict[str, float]:
    det_yes = det_total = false_yes = false_total = 0
    iou_sum = iou_cnt = 0
    p_sum = p_cnt = r_sum = r_cnt = 0
    all_sum = all_cnt = 0

    for r in all_records:
        if r["cue_match"] is not None:
            det_total += 1
            if r["cue_match"] == "YES":
                det_yes += 1
        if r["false_match"] is not None:
            false_total += 1
            if r["false_match"] == "YES":
                false_yes += 1
        if r["iou"] is not None:
            iou_cnt += 1
            iou_sum += r["iou"]
        if r["p_sub_accuracy"] is not None:
            p_cnt += 1
            p_sum += r["p_sub_accuracy"]
        if r["r_sub_accuracy"] is not None:
            r_cnt += 1
            r_sum += r["r_sub_accuracy"]
        if r["all_sub_correct"] is not None:
            all_cnt += 1
            all_sum += r["all_sub_correct"]

    return {
        "det_accuracy": det_yes / det_total if det_total else None,
        # "false_accuracy": false_yes / false_total if false_total else None,
        "iou_avg": iou_sum / iou_cnt if iou_cnt else None,
        "p_sub_acc_avg": p_sum / p_cnt if p_cnt else None,
        "r_sub_acc_avg": r_sum / r_cnt if r_cnt else None,
        "all_sub_acc": all_sum / all_cnt if all_cnt else None,
        "false_accuracy": false_yes / false_total if false_total else None,
    }


# ────────────────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Pass --api_key or set env var OPENAI_API_KEY.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    records = parse_dataset(Path(args.input_file))
    print(f"→ Loaded {len(records)} records.")

    # Only the pretty JSON output is kept. The JSONL streaming output has been removed.
    pretty_json_path = Path(args.output_file).with_suffix(".json")

    all_out = []
    with tqdm(total=len(records), desc="Processing") as bar:
        for rec in records:
            has_cue = rec["label"]

            # ── Deterministic-cue branch ──────────────────────────────────
            if has_cue:
                raw_output = call_gpt(client, args.gpt_model, rec["answer"])
                summary_items = (
                    [] if "no evidence found" in raw_output.lower()
                    else split_numbered(raw_output)
                )

                cue_result = judge_cue(client, args.gpt_model, rec["detm_cue"], raw_output)

                iou_count, iou_val = intersection_iou(
                    client, args.gpt_model, rec["non_detm_cue"], summary_items
                )

                p_correct, p_acc = evaluate_sub_accuracy(
                    client, args.gpt_model,
                    rec["p_sub_ins"], rec["p_sub_ans_model"], rec["p_sub_ans_ref"]
                )
                r_correct, r_acc = evaluate_sub_accuracy(
                    client, args.gpt_model,
                    rec["r_sub_ins"], rec["r_sub_ans_model"], rec["r_sub_ans_ref"]
                )
                all_flag = int(
                    (p_acc == 1.0 if p_acc is not None else True) and
                    (r_acc == 1.0 if r_acc is not None else True)
                )

                false_match = None

            # ── False-image branch ────────────────────────────────────────
            else:
                raw_output = ""
                summary_items = []
                cue_result = None
                iou_val = None

                p_correct = p_acc = r_correct = r_acc = all_flag = None

                false_match = verify_false_image(client, args.gpt_model, rec["answer"])

            # ── Build output dict ─────────────────────────────────────────
            out_rec = {
                "data_path": rec["data_path"],
                "main_ins": rec["main_ins"],
                "scene": rec["scene"],
                "target": rec["target"],
                "label": rec["label"],
                "detm_cue": rec["detm_cue"],
                "non_detm_cue": rec["non_detm_cue"],
                "answer": rec["answer"],
                "summarize": raw_output,
                "summarize_list": summary_items,
                "cue_match": cue_result,
                "iou": iou_val,
                "inter_cont": iou_count,
                "p_sub_ins": rec["p_sub_ins"],
                "p_sub_ans": rec["p_sub_ans_model"],
                "p_sub_accuracy": p_acc,
                "r_sub_ins": rec["r_sub_ins"],
                "r_sub_ans": rec["r_sub_ans_model"],
                "r_sub_accuracy": r_acc,
                "all_sub_correct": all_flag,
                "false_match": false_match,
            }

            # Collect results; JSONL file writing has been removed.
            all_out.append(out_rec)
            bar.update(1)

    # save pretty JSON for quick view
    with pretty_json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)

    # print overall metrics
    overall = final_metrics(all_out)
    print("\n===== FINAL METRICS =====")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
    print(f"\nFile written:\n  • {pretty_json_path}")


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process v2 dataset and evaluate cues / sub-questions."
    )
    parser.add_argument("--input_file", required=True, help="Source JSON dataset")
    parser.add_argument("--output_file", required=True,
                        help="Basename for output (writes .json)")
    parser.add_argument("--gpt_model", default="gpt-4.1", help="Chat model name")
    parser.add_argument("--api_key", help="OpenAI API key")
    main(parser.parse_args())
