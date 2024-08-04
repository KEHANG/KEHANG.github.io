---
layout: post
title:  "Battle-Tested LLM Training: The Input Pipeline"
date:   2024-07-24 10:16:16
tags: llm
---

I recently discovered that the [MaxText](https://github.com/google/maxtext) project serves as an excellent reference resource for learning about the latest developments in LLM training and inference. The README advertises it as a high-performance, highly scalable solution that achieves high Model FLOPs Utilization (MFUs) and is compatible with both TPUs and GPUs (for which we should likely thank the Jax team). If these claims are true, which I believe they are, this codebase should attract practitioners, researchers, and students to explore it and learn the best and latest practices.

This brief post focuses on the input pipeline in a multi-host setting. You can roughly think of multi-host as running programs over multiple GPU hosts or TPU hosts (each host has one or more workers called devices). The input pipeline performs at two stages:

- stage 1: it distributes the dataset across all hosts, i.e., each host gets a shard of the dataset.
- stage 2: at the local dataset shard, during iteration, it distributes the batch across the accelerator devices under the host (each TPUv5e host has 4 accelerator devices, while each GPU host typically has 1 accelerator device).

How do we achieve that in code? Since [grain](https://github.com/google/grain) is becoming a popular choice (see [comparison to TFDS and HuggingFace](https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md)) in the Jax world, let's dive into a grain implementation of such an input pipeline: [preprocessing_pipeline](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/input_pipeline/_grain_data_processing.py#L38). More concretely, let's say we want to train a model with batch size 512 using 256 TPUv5e chips, i.e., 64 hosts.

- The first question is what batch size each shard should have, in our example, since global batch size is 512, each host is responsible for one 64th, so the local shard batch size is 8. ([shard batch size](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/input_pipeline/_grain_data_processing.py#L78)): 
```
batch_size=global_batch_size // jax.process_count()
```
- Then next question is which host gets which 64th shard of the dataset. This is specified by `grain.IndexSampler` and its `shard_options` argument ([code-link](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/input_pipeline/_grain_data_processing.py#L84-L92)):
```
shard_options=grain.ShardOptions(
    shard_index=dataloading_host_index, shard_count=dataloading_host_count
)
```
For the first host, `shard_index` is 0, and `shard_count` is 64.

- Now for stage 2, each host will be distributing local batches of size 8 across 4 devices. MaxText has this iterator class [MultiHostDataLoadIterator](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/multihost_dataloading.py#L93), which takes in a dataloader and turns it into an iterator ([local_iterator](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/multihost_dataloading.py#L102C29-L102C50)), and its [\_\_next\_\_ method](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/multihost_dataloading.py#L119) will do the batch distribution.

- The actual heavy lifting (distributing) is done by [_form_global_array](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/multihost_dataloading.py#L50C5-L50C23). First, it [splits](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/multihost_dataloading.py#L55) the batch array into N pieces (N is the number of local devices) and then [put each piece to assigned device](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/multihost_dataloading.py#L63) in order. Finally, it informs jax that those local arrays form a global array (this becomes more relevant when we talk about `pjit` in future posts.)

Some final thought, it's not always the case to distribute the host batch across all local devices; it depends on [`data_sharding` configuration](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/configs/base.yml#L203). We'll probably dive deeper into this later, for now, distributing all the way to each local device is a good starting point.

*if you have comments or suggestions or spotted an error, please let me know via email: kehanghan at gmail dot com.*