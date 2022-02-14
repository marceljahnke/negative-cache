# coding=utf-8
# Copyright 2022 Marcel Jahnke
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist

"""Utilities for negative cache training."""


def approximate_top_k_with_indices(negative_scores, k):
    """Approximately mines the top k highest scoreing negatives with indices.

    This function groups the negative scores into num_negatives / k groupings and
    returns the highest scoring element from each group. It also returns the index
    where the selected elements were found in the score matrix.

    Args:
      negative_scores: A matrix with the scores of the negative elements.
      k: The number of negatives to mine.

    Returns:
      The tuple (top_k_scores, top_k_indices), where top_k_indices describes the
      index of the mined elements in the given score matrix.
    """
    bs = negative_scores.size()[0]
    num_elem = negative_scores.size()[1]
    device = negative_scores.device

    batch_indices = torch.arange(end=num_elem).detach()
    # batch_indices --> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, ..., cache_size - 1])
    # torch.unsqueeze(batch_indices, dim=0) --> tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, ..., cache_size - 1]])
    indices = torch.tile(torch.unsqueeze(batch_indices, dim=0), dims=(bs, 1)).detach()
    # indices: tensor([
    #   [0, 1, 2, 3, 4, 5, 6, 7, 8, ..., cache_size - 1], <-- 0
    #   ...
    #   [0, 1, 2, 3, 4, 5, 6, 7, 8, ..., cache_size - 1]  <-- batch_size - 1
    # ])
    # indices: [batch_size x cache_size]

    # negative_scores: [batch_size x cache_size]
    # grouped_negative_scores: [batch_size * top_k x cache_size / k]
    grouped_negative_scores = torch.reshape(negative_scores, (bs * k, -1)).detach()

    grouped_batch_indices = (
        torch.arange(end=grouped_negative_scores.size()[0]).to(device).detach()
    )
    # grouped_batch_indices: [0, 1, ..., batch_size * top_k - 1] --> size [batch_size * top_k]
    # with indices for rows in grouped_negative_scores

    grouped_top_k_scores, grouped_top_k_indices = torch.topk(
        grouped_negative_scores, k=1
    )
    # grouped_top_k_scores and grouped_top_k_indices: [batch_size * top_k x 1] each

    grouped_top_k_indices = torch.squeeze(grouped_top_k_indices, dim=1).detach()
    # grouped_top_k_indices: [batch_size * top_k x 1] --> [batch_size * top_k]

    gather_indices = torch.stack(
        [grouped_batch_indices, grouped_top_k_indices], dim=1
    ).detach()
    # gather_indices: [batch_size * top_k x k=1]

    grouped_indices = torch.reshape(indices, (bs * k, -1)).detach()
    # grouped_indices: [batch_size * top_k x cache_size / top_k]

    grouped_top_k_indices = gather_nd(grouped_indices, gather_indices).detach()
    # grouped_top_k_indices: [top_k * batch_size] ==> indices for top1 of each grouped_negative_scores
    # for all batches converted to original indices of negative_scores

    top_k_indices = torch.reshape(grouped_top_k_indices, (bs, k)).detach()
    top_k_scores = torch.reshape(grouped_top_k_scores, (bs, k)).detach()
    # top_k_indices, top_k_scores: [batch_size x top_k] each
    # containing the indices of the top_k scores (from the cache between current query emb and cache emb);
    # each row contains values for a query (number of rows == batch_size)
    # each indice or score represents the top1 score from the respective batch_size * top_k grouped
    # negative_score ==> only approximated top_k from negative_scores ungrouped

    return top_k_scores, top_k_indices


def cross_replica_concat(tensor: torch.Tensor, group=None):
    with torch.cuda.device(dist.get_rank()):
        tensor_list = [
            torch.empty_like(tensor).detach().cuda()
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(tensor_list, tensor.contiguous(), group)
        tensor_concat = torch.concat(tensor_list).detach().cuda()
        del tensor_list
    return tensor_concat


def gather_nd(params, indices):
    """
    source: https://discuss.pytorch.org/t/implement-tf-gather-nd-in-pytorch/37502/7

    The input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    """
    # Normalize indices values
    params_size = list(params.size())

    assert len(indices.size()) == 2
    assert len(params_size) >= indices.size(1)
    del params_size

    # Generate indices
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).detach().long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    params = params.reshape(
        (-1, *tuple(torch.tensor(params.size()[ndim:]).detach()))
    ).detach()
    return params[idx]


def tensor_update(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    padding=False,
    padding_token=0,
) -> torch.Tensor:
    """
    Updates values of `tensor` at `indices` with values of `updates`.

    Args:
        tensor ([type]): Tensor to be updated.
        indices ([type]): Tensor containing indices at which `tensor` will be updated.
        updates ([type]): Tensor containing updates, which will be inserted at `indices` into `tensor`.
            Number of updates must match number of indices.

    Returns:
        torch.Tensor: updated tensor
    """
    # tensors in updates represent new document features (e.g. tensors for input_ids, attention_mask),
    # embedding tensors or age tensors -> new rows in those tensors
    # the new rows for input_ids, attention_mask and embeddings can be shorter due to now reaching the maximum token size.
    # since this function is not as refined as tf.tensor_scatter_nd_update() those tensors must be padded
    # original project was for TPUs which need batches of the same size -> size of update was always the same as to be
    # updated value in tensor

    # dimensional constraints from tf.tensor_scatter_nd_update()
    assert indices.dim() >= 2  # list of indices
    index_depth = indices.shape[-1]
    assert (
        index_depth <= tensor.dim()
    )  # scalar update if index_depth == tensor.dim else slice update
    slice_update = index_depth < tensor.dim()

    updates = updates.type(tensor.dtype)
    indices = indices.long()  # torch.tensor.index_put_ constraint

    for i, idx in enumerate(indices):
        update = updates[i]

        if padding and slice_update:
            diff = len(tensor[0]) - len(update)
            pad = padding_token * torch.ones(
                size=[diff], dtype=update.dtype
            ).detach().to(updates.device)
            update = torch.cat([update, pad]).detach()

        tensor[idx] = update.to(tensor.device)
    del update
    return tensor
