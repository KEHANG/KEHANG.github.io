---
layout: post
title: "Battle-Tested LLM Training: From Dataset to Data Iterator"
date: 2024-08-11 10:16:16
tags: llm
---

If you find an interesting dataset (often from either Huggingface or TFDS nowadays) and you'd like to use it for LLM training, this post is for you! Specifically, I'll be explaining the process that gradually turns a Huggingface dataset to an iterator that's ready to feed model training with batches of data. Conceptually it takes four steps.

![Alt text]({{ site.github.url }}/assets/battle_tested_llm_training/input_pipeline_data2iter.png){:width="85%"}

To make it concrete, I'll use MaxText's [make_hf_iterator](https://github.com/google/maxtext/blob/da50760ac0baf3920305a365215f6f0c5f110ad2/MaxText/input_pipeline/_hf_data_processing.py#L121) as my reference code, and choose [openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k) as our input dataset.

### load raw dataset

First, let's load the raw `openwebtext-10k` dataset. If `streaming` is on as shown below, data files will not be downloaded. Instead, it streams the data progressively while iterating on the dataset.
```python
from datasets import load_dataset

dataset = load_dataset("stas/openwebtext-10k", split="train", streaming=True)
```

### tokenize
At this stage, we need tokenize the raw dataset's `text` field and trim the tokenized sequence up to predefined `max_length`. Practically, we'd first create a tokenizer either from a local file or from Huggingface via `tokenizer_path`. In the example below, we use `t5-small` tokenizer, which would be fetched from Huggingface directly.

```python
# Sets some constants
add_bos, add_eos, max_length = True, True, 512
tokenizer_path = "t5-small"
data_column_name = "text"

# Creates a tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    tokenizer_path,
    add_bos_token=add_bos,
    add_eos_token=add_eos,
    model_max_length=max_length,
    legacy=False,
)
```
Tokenization is then accomplished by running that tokenizer via dataset `map` function, [_input_pipeline_utils.tokenization](https://github.com/google/maxtext/blob/ed21f6ad1bc60285958d585753dda01e1ddfa664/MaxText/input_pipeline/_input_pipeline_utils.py#L56). This function applies the above tokenizer to the field `data_column_name` of each data example and truncates the tokens up to `max_length`.
```python
from maxtext.MaxText.input_pipeline import _input_pipeline_utils

dataset = dataset.map(
    _input_pipeline_utils.tokenization,
    batched=True,
    fn_kwargs={"hf_tokenizer": tokenizer, "max_length": max_length - 1, "column_name": data_column_name},
)
# Post-tokenization: renaming the field where the tokens are.
dataset = dataset.select_columns(["input_ids"]).rename_column("input_ids", data_column_name)
```

### transform: pack and shift
After tokenization, data examples become token sequences of various lengths. To increase training efficiency, we try to pack as many sequences as possible into the context window (`max_length` in the code). Here we use grain's experimental packing API [PackAndBatchOperation](https://github.com/google/grain/blob/9b984e8a2ccdd3d377cb43b17591080b44c07009/grain/_src/python/experimental/example_packing/packing.py#L152).

In multi-host setting, each host (i.e., process) gets an equal share of the global batch size (say `512`), and this input pipeline code runs at host-level in parallel, thus we want host-level batch size when we batch, i.e., `global_batch_size // jax.process_count()`.

```python
import grain.python as grain
# Sets some constants.
global_batch_size = 512

# Adds packing transformation.
transformations = []
# HFNormalizeFeatures makes two copies of `text` field: one is called
# `inputs` and the other `targets`.
operations.append(_input_pipeline_utils.HFNormalizeFeatures(data_column_name))
transformations.append(
    grain.experimental.PackAndBatchOperation(
        # In multi-host setting, each host (i.e., process) gets an equal share 
        # of the global batch size.
        # And this input pipeline runs at host-level in parallel, thus we want 
        # host-level batch size here.
        batch_size=global_batch_size // jax.process_count(),
        length_struct={"inputs": max_length, "targets": max_length},
    )
)

# Post-packing: reformating tuple to flat dict style.
transformations.append(_input_pipeline_utils.ReformatPacking())
```
Finally we shift the `inputs` field by 1 token to the right, to make it ready for teacher-forcing training.
```python
transformations.append(_input_pipeline_utils.ShiftData(axis=1))
```

### sample
Now with all the transformations done, we need to tell each host how to sample from the transformed dataset. Most common settings include number of epochs (`num_epochs`), which shard of the dataset the current host should load (`shard_options`), whether to shuffle (`shuffle`), etc.

```python
sampler = grain.IndexSampler(
    num_records=len(dataset),
    num_epochs=1,
    shard_options=grain.ShardOptions(
        shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
    ),
    shuffle=False,
    seed=0,
)
```

### put together
We put everything together with `grain.DataLoader` API, which takes in the raw `dataset`, training-required `transformations` and `sampler`. The returned dataloader is ready to produce batches the downstream training loop needs (`iter(dataloader)`).

```python
dataloader = grain.DataLoader(
    data_source=dataset,
    operations=transformations,
    sampler=sampler,
    worker_count=1,
    worker_buffer_size=1,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
)

data_iter = iter(dataloader)
batch = next(data_iter)
```

### Final words
Feel free to run and fork [input_pipeline_data2iter.ipynb](https://colab.research.google.com/drive/1MrIvDAiWcTSma3mDKwmd5F_cxVznfxmQ#scrollTo=VbMbtALpE7om) if you'd like to run a complete version of input pipeline. It's worth noting that the returned `batch` sits in host CPU memory and so it's necessary to further shard it across TPU devices before feeding the batch to `pjitted` train step. This could be done by [MultiHostDataLoadIterator](https://github.com/google/maxtext/blob/ead18fbe6f2d8a6cbae6bbd38568146919e20e18/MaxText/multihost_dataloading.py#L93). If you'd like to know the details, [this previous post]({% post_url 2024-07-24-battle-tested-llm-training-multihost-input-pipeline %}) could be of interest. If you'd like to run the input pipeline