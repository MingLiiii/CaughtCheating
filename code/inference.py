"""
python gpt4o_infer_standalone.py \
  --root_dir c_bench_copy \
  --save_dir vqa_outputs \
  --datatype json \
 """


import os
import sys
import json
import copy
import time
import re
import argparse
import base64
from io import BytesIO
from typing import List, Dict

from openai import OpenAI
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utility functions previously in utils.path_utils
# -----------------------------------------------------------------------------

def set_root_folder():
    """Return folders used for caching and data.

    This is a simplified version of utils.path_utils.set_root_folder that
    chooses reasonable defaults so the script can run anywhere without
    editing absolute paths.  Feel free to modify these paths to match your
    local environment.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    root_dir = os.getcwd()  # default to current working directory
    image_folder = root_dir
    return cache_dir, root_dir, image_folder


# Apply the cache folders so transformers / datasets do not complain
CACHE_DIR, ROOT_DIR, _ = set_root_folder()
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# -----------------------------------------------------------------------------
# Utility functions previously in utils.infer_utils
# -----------------------------------------------------------------------------

def load_data(data_type: str, root_dir: str = "") -> List[Dict]:
    """Load evaluation data from a JSONL file.

    The function expects a file named ``data_info.jsonl`` inside *root_dir*.
    Each line is parsed as JSON.  For every entry that contains ``main_ins`` it
    builds a list of questions combining ``main_ins`` with optional
    ``p_sub_ins`` and ``r_sub_ins`` fields.  It also attaches an absolute image
    path (``img_path``) so downstream code can open the image directly.
    """
    list_eval: List[Dict] = []

    file_path = os.path.join(root_dir, "data_info.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find data_info.jsonl at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            if "main_ins" not in data:
                # skip entries without the mandatory question
                continue

            # Absolute path to the image
            data["img_path"] = os.path.join(root_dir, data["data_path"]) if "data_path" in data else None

            # Build list of (question_type, question_text)
            list_questions = [["main_ins", data["main_ins"]]]
            if "p_sub_ins" in data:
                list_questions.extend([["p_sub_ins", q] for q in data["p_sub_ins"]])
            if "r_sub_ins" in data:
                list_questions.extend([["r_sub_ins", q] for q in data["r_sub_ins"]])
            data["list_questions"] = list_questions

            list_eval.append(data)

    return list_eval


# -----------------------------------------------------------------------------
# Functions from model/gpt4o_infer.py (slightly refactored for standalone use)
# -----------------------------------------------------------------------------

def encode_image(datatype: str, data: Dict) -> str:
    """Encode the image referenced by *data* to a base64 JPEG string."""
    if datatype != "json":
        # Expect a PIL Image object already loaded in data["image"]
        image = data["image"].convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        # Expect a file path in data["img_path"]
        if not os.path.exists(data["img_path"]):
            raise FileNotFoundError(f"Image not found at {data['img_path']}")
        with open(data["img_path"], "rb") as image_file:
            img_str = base64.b64encode(image_file.read()).decode("utf-8")

    return img_str


def ask_gpt4o_about_image(
    api_key: str,
    datatype: str,
    data: Dict,
    model_name: str = "gpt-4o",
    max_retries: int = 3,
):
    """Send an image & a list of questions to GPT-4o and return answers.

    Parameters
    ----------
    api_key: str
        Your OpenAI API key.
    datatype: str
        Either "json" (expects ``img_path``) or a different value meaning the
        image is already loaded as a PIL Image in ``data['image']``.
    data: dict
        A single entry of the dataset containing ``list_questions``.
    max_retries: int, default 3
        Number of times to retry the request if the OpenAI API errors.
    """
    base64_image = encode_image(datatype, data)

    client = OpenAI(api_key=api_key)

    answers = []
    for qs_type, question in data["list_questions"]:
        attempt = 0
        model_answer = None
        while attempt < max_retries and model_answer is None:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                )
                model_answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Warning] Error on attempt {attempt + 1}: {e}")
                time.sleep(2)
                attempt += 1

        answers.append([qs_type, question, model_answer])
    return answers


def process_dataset(
    api_key: str,
    datatype: str,
    eval_dataset: List[Dict],
    save_dir: str,
    model_name: str = "gpt-4o",
):
    """Iterate over the dataset, query GPT-4o and save answers as JSON."""
    results: Dict[int, Dict] = {}

    for idx, data in enumerate(tqdm(eval_dataset, desc="Processing")):
        answers = ask_gpt4o_about_image(api_key, datatype, data, model_name=model_name)

        entry = copy.deepcopy(data)
        # No need to keep the question list in the final JSON
        entry.pop("list_questions", None)
        entry["answers"] = answers
        results[idx] = entry

    output_path = save_dir
    print(f"[Info] Writing results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visual question answering with GPT-4o (standalone script)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./dataset",
        help="Directory that contains data_info.jsonl and the images",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="gpt4o_results",
        help="Where to save the JSON results",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="json",
        help="Indicate whether images are referenced by path (json) or passed as PIL objects",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        required=True,
        help="Your OpenAI API key",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="OpenAI model name to query (e.g. gpt-4o, gpt-4o-mini)",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load data & run inference
    # ---------------------------------------------------------------------
    dataset = load_data(data_type=args.datatype, root_dir=args.root_dir)
    process_dataset(
        api_key=args.api_key,
        datatype=args.datatype,
        eval_dataset=dataset,
        save_dir=args.save_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main() 
