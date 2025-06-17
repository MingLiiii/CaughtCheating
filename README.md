# CaughtCheating: Are Current MLLMs Good Cheating Detectives? Towards the Boundary of Visual Perception and Reasoning

[CaughtCheating: Are Current MLLMs Good Cheating Detectives? Towards the Boundary of Visual Perception and Reasoning](https://github.com/MingLiiii/CaughtCheating)

  This is the repo for the CaughtCheating project, in which we explore the current boundary of MLLMs and construct a hard benchmark for visual perception and reaasoning.  

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
- [Install](#install)
- [Run Code](#run-code)
- [Data](#data)
- [ToDo](#todo)
- [Citation](#citation)
- [Our Related Works](#our-related-works)

## Overview

What makes a difference in the post-training of LLMs? We investigate the training patterns of different layers in large language models (LLMs), through the lens of gradient, when training with different responses and initial models. We are specifically interested in how fast vs. slow thinking affects the layer-wise gradients, given the recent popularity of training LLMs on reasoning paths such as chain-of-thoughts (CoT) and process rewards. In our study, fast thinking without CoT leads to larger gradients and larger differences of gradients across layers than slow thinking (Detailed CoT), indicating the learning stability brought by the latter. Moreover, pre-trained LLMs are less affected by the instability of fast thinking than instruction-tuned LLMs. Additionally, we study whether the gradient patterns can reflect the correctness of responses when training different LLMs using slow vs. fast thinking paths. The results show that the gradients of slow thinking can distinguish correct and irrelevant reasoning paths. As a comparison, we conduct similar gradient analyses on non-reasoning knowledge learning tasks, on which, however, trivially increasing the response length does not lead to similar behaviors of slow thinking. Our study strengthens fundamental understandings of LLM training and sheds novel insights on its efficiency and stability, which pave the way towards building a generalizable System-2 agent.

## Highlights

**Our Key Findings:**
* Training LLMs for slow thinking (Detailed CoT) leads to similar gradient norms of different layers, while fast thinking (Simplified/None CoT) results in larger gradients (fast forgetting) of earlier layers and drastic differences across layers.
* The gradient of slow thinking (Detailed CoT) helps distinguish correct responses from irrelevant responses. Without CoT, the gradient patterns of the two types of responses are similar.
* The instruction-finetuned LLMs do not show superior capability over pre-trained base LLMs in identifying incorrect reasoning paths.
* The above observations on reasoning tasks (math and commonsense) cannot be extended to knowledge learning tasks, where simply increasing response length does not show similar gradient patterns as slow thinking.

## Run Code

TBD

## Data

The data con be found in the folder `CaughtCheating_data`.

## Citation

Please consider citing our papers if you think our code or data are useful. Thank you! <br>
```

```




