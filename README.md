# CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning

[CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning](https://arxiv.org/abs/2507.00045)

  This is the repo for the CaughtCheating project, in which we explore the current boundary of MLLMs and construct a hard benchmark for visual perception and reasoning.  

<p align="center" width="40%">
<a ><img src="images/logo.png" alt="overview" style="width: 40%; min-width: 300px; display: block; margin: auto;"></a>
</p>

The repo contains:

- The data for CaughtCheating Benchmark.
- The code for CaughtCheating Evaluation.

(Feel free to email Ming ([Email](minglii@umd.edu)) for any questions or feedback.)

## News
- [2025/06] We released our project. 

## Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Run Code](#run-code)
- [Data](#data)
- [ToDo](#todo)
- [Citation](#citation)
- [Our Related Works](#our-related-works)

## Overview

Recent agentic Multi-Modal Large Language Models (MLLMs) such as GPT-o3 have achieved near-ceiling scores on various existing benchmarks, motivating a demand for more challenging test tasks. These MLLMs have been reported to excel in a few expert-level tasks for humans, e.g., GeoGuesser, reflecting their potential as a detective who can notice minuscule cues in an image and weave them into coherent, situational explanations, leading to a reliable answer. But can they match the performance of excellent human detectives? To answer this question, we investigate some hard scenarios where GPT-o3 can still handle, and find a common scenario where o3's performance drops to nearly zero, which we name CaughtCheating. It is inspired by the social media requests that ask others to detect suspicious clues from photos shared by the poster's partner. We conduct extensive experiments and analysis to understand why existing MLLMs lack sufficient capability to solve this kind of task. CaughtCheating provides a class of challenging visual perception and reasoning tasks with great value and practical usage. Success in these tasks paves the way for MLLMs to acquire detective-level visual perception and reasoning capabilities.

## Highlights

* We systematically evaluate the limits of current MLLMs in visual perception and reasoning, analyzing how they solve various complex tasks via sophisticated reasoning strategies, and **identify a common scenario where even advanced models like o3’s performance drops to nearly zero**.
* We present **CaughtCheating**, the first benchmark specifically designed to **assess the ability to actively search and detect subtle, context-dependent suspicious clues in real-world images**. Most human annotators and state-of-the-art agentic MLLMs struggle to succeed on CaughtCheating tasks, highlighting the lack of detective-level exploration skills.
* We analyze why even the most advanced agentic MLLMs fail on CaughtCheating. Inspired by the **Guided Search** theory, we find that these models often **lack awareness of what to search for and how to relate observed details to the query**. Our findings offer insights into both the construction of more challenging benchmarks and the limitations of existing MLLMs.

## Run Code

TBD

## Data

The data can be found in the folder `data`.

## Citation

Please consider citing our papers if you think our code or data are useful. Thank you! <br>
```
@article{li2025caughtcheating,
  title={CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning},
  author={Li, Ming and Wang, Chenguang and Liang, Yijun and Wang, Xiyao and Zhou, Yuhang and Wu, Xiyang and Zhang, Yuqing and Zhang, Ruiyi and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2507.00045},
  year={2025}
}
```




