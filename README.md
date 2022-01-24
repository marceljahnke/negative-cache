# Efficient Training of Retrieval Models using Negative Cache

![Tests](https://github.com/marceljahnke/negative-cache/actions/workflows/lint-and-test.yml/badge.svg)

This repository contains a PyTorch implementation of the paper [Efficient Training of Retrieval Models using Negative Cache](https://openreview.net/pdf?id=824xC-SgWgU). It's a training approach for a dual encoder, that uses a memory efficient negative streaming cache.

The general idea, according to the authors, is to sample negatives from the cache and use them in combination with GumbelMax-sampling to approximate the cross-entropy loss function, at each iteration. By design the cache can store a large amount of negatives in a memory efficient way.

The original implementation can be found [here](https://github.com/google-research/google-research/tree/master/negative_cache).


---
- [Efficient Training of Retrieval Models using Negative Cache](#efficient-training-of-retrieval-models-using-negative-cache)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Special cases](#special-cases)
  - [DistributedDataParallel usage](#distributeddataparallel-usage)
---

## Installation

To install, run the following commands:

```bash
git clone git@github.com:marceljahnke/negative-cache.git
cd negative-cache
python -m pip install .
``` 

You can also install the package in editable mode:

```bash
python -m pip install -e .
```

## Usage

The generel usage is very similair to the tensorflow version. This chapter explains the usage on a single GPU, see [DistributedDataParallel usage](#distributeddataparallel-usage) for multi-gpu usage.

Set up the specs that describe the document feature dictionary. These describe the feature keys and shapes for the items we need to cache. The document features represent the features that are used to compute the embedding by using the `document_network`.

```python
data_keys = ('document_feature_1', 'document_feature_2')
embedding_key = 'embedding'
specs = {
    'document_feature_1': tf.io.FixedLenFeature([document_feature_1_size], tf.int32),
    'document_feature_2': tf.io.FixedLenFeature([document_feature_2_size], tf.int32),
    'embedding': tf.io.FixedLenFeature([embedding_size], tf.float32)
}
```

Set up the cache loss.

```python
cache_manager = negative_cache.CacheManager(specs, cache_size=131072)
cache_loss = losses.CacheClassificationLoss(
    embedding_key=embedding_key,
    data_keys=data_keys,
    score_transform=lambda score: 20.0 * score,  # Optional, applied to scores before loss.
    top_k=64  # Optional, restricts returned elements to the top_k highest scores.
)
handler = handlers.CacheLossHandler(
    cache_manager, cache_loss, embedding_key=embedding_key, data_keys=data_keys)
```

Calculate the cache loss using your query and document networks and data.

```python
query_embeddings = query_network(query_data)
document_embeddings = document_network(document_data)
loss = handler.update_cache_and_compute_loss(
    document_network, 
    query_embeddings,
    document_embeddings, 
    document_data,
    writer # Optional, used to log additional information to tensorboard.
    )
```

You can call the handler with an optional [`torch.utils.tensorboard.SummaryWriter`](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter) to log additional information, i.e. interpretable loss and staleness of the cache.


## Special cases

If your document features consists of only one feature, pass it as a tuple containing only one item:
```python
data_keys = ('document_feature_1',)
```

## DistributedDataParallel usage

When using [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) do not use a lambda function for the score_transform, instead write a regular function. This way the function is pickleable.

```python
def fn(scores):
    return 20 * scores

...
score_transform=fn
```
instead of

```python
score_transform=lambda scores: 20.0 * scores,
```

You also want to use the `DistributedCacheClassificationLoss` instead of the `CacheClassificationLoss`:

```python
cache_loss = DistributedCacheClassificationLoss(
            embedding_key=embedding_key,
            data_keys=data_keys,
            score_transform=score_transformation,
            top_k=top_k,
        )
```