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

"""Implements loss functions for dual encoder training with a cache."""

import abc
import collections
import torch
import torch.nn.functional as F
import torch.distributed as dist

from negative_cache import retrieval_fns
from negative_cache import util

CacheLossReturn = collections.namedtuple(
    "CacheLossReturn",
    [
        "training_loss",
        "interpretable_loss",
        "updated_item_data",
        "updated_item_indices",
        "updated_item_mask",
        "staleness",
        "topk_features",
    ],
)


class CacheLoss(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self, doc_network, query_embeddings, pos_doc_embeddings, cache, return_topk
    ):
        pass


_RetrievalReturn = collections.namedtuple(
    "_RetrievalReturn",
    [
        "retrieved_data",
        "scores",
        "retrieved_indices",
        "retrieved_cache_embeddings",
        "retrieved_cache_topk_features",
    ],
)


def _score_documents(
    query_embeddings, doc_embeddings, score_transform=None, all_pairs=False
):
    """Calculates the dot product of query, document embedding pairs."""
    if all_pairs:
        scores = torch.matmul(query_embeddings, doc_embeddings.T)
    else:
        scores = torch.sum(query_embeddings * doc_embeddings, dim=1)
    if score_transform is not None:
        scores = score_transform(scores)
    return scores


def _batch_concat_with_no_op(tensors):
    """If there is only one tensor to concatenate, this is a no-op."""
    if len(tensors) == 1:
        return tensors[0].detach()
    else:
        return torch.concat(tensors, dim=0).detach()


def _retrieve_from_caches(
    query_embeddings,
    cache,
    retrieval_fn,
    embedding_key,
    data_keys,
    sorted_data_sources,
    return_topk=0,
    score_transform=None,
    top_k=None,
):
    """Retrieve elements from a cache with the given retrieval function."""
    device = query_embeddings.device
    # all_embeddings: Tensor with all embeddings as rows
    all_embeddings = _batch_concat_with_no_op(
        [cache[data_source].data[embedding_key] for data_source in sorted_data_sources]
    ).detach()
    all_data = {}
    # all_data: dict with same keys as specs (without embeddings) and Tensors as valules.
    # Tensors have all features from specs as rows.
    for key in data_keys:
        all_data[key] = _batch_concat_with_no_op(
            [cache[data_source].data[key] for data_source in sorted_data_sources]
        ).detach()
    # scores: query embedding from current batch * all_embeddings from cache -->
    # scores for all queries from batch and document embedding from cache

    # query_embeddings: [batch_size x 768 (BERT CLS emb)] * all_embeddings: [cache_size x 768 (BERT CLS emb)].T
    # --> scores: [batch_size x cache_size]
    scores = _score_documents(
        query_embeddings,
        all_embeddings,
        score_transform=score_transform,
        all_pairs=True,
    ).detach()

    retrieved_topk_features = None
    if top_k:
        scores, top_k_indices = util.approximate_top_k_with_indices(scores, top_k)
        # scores, top_k_indices: [batch_size x top_k]
        top_k_indices = top_k_indices.type(torch.int64).detach().to(device)
        scores = (
            scores.detach()
        )  # scores are only used to identify the current 'best' negative documents in the cache

        if return_topk > 1:
            # this features are not gumbel-max sampled
            n = min(return_topk, top_k_indices.shape[-1])
            # first we need to find the indices of the n highest scores from the topk
            # retrieved elements
            _, top_n_scores_indices = torch.topk(scores, n)
            idx = (
                top_n_scores_indices.view(-1).unsqueeze(dim=1).detach().to(device)
            )  # [B*n x 1]
            batch = (
                (
                    torch.arange(top_n_scores_indices.shape[0])
                    .unsqueeze(dim=-1)
                    .tile(top_n_scores_indices.shape[-1])
                    .view(-1)
                    .unsqueeze(dim=-1)
                )
                .detach()
                .to(device)
            )  # [B*n x 1]
            indices = torch.cat((batch, idx), dim=1).detach().to(device)  # [B*n x 2]
            top_n_indices = (
                util.gather_nd(top_k_indices, indices)
                .view(top_k_indices.shape[0], -1)
                .detach()
            )  # [B x n]
            retrieved_topk_features = [
                {
                    k: util.gather_nd(
                        v.detach(),
                        top_n_indices[b].unsqueeze(dim=1).to(torch.device("cpu")),
                    ).detach()
                    for k, v in all_data.items()
                }
                for b in range(top_n_indices.shape[0])
            ]  # [B x {str: [n x feature_size]}]
            del top_n_scores_indices, idx, batch, indices, top_n_indices

        retrieved_indices = retrieval_fn(scores)
        retrieved_indices = retrieved_indices.detach()
        # ONE index for each query! extracted from the top_k for each query
        # retrieved_indices contains index of highest score in cache after modification with
        # gumbel value (and inv_temp) for each query out of top_k values
        # retrieved_indices: [batch_size x 1]

        batch_index = (
            torch.unsqueeze(
                torch.arange(
                    end=retrieved_indices.size()[0], dtype=torch.int64
                ).detach(),
                dim=1,
            )
            .detach()
            .to(device)
        )
        # batch_index: [batch_size x 1]
        retrieved_indices_with_batch_index = (
            torch.concat([batch_index, retrieved_indices], dim=1).detach().to(device)
        )
        # retrieved_indices_with_batch_index: [batch_size x 1 + 1] == [batch_size x 2]

        retrieved_indices = util.gather_nd(
            top_k_indices, retrieved_indices_with_batch_index
        ).detach()
        # retrieved_indices: [batch_size] == converted through retrieval function (GumbelMax) retrieved
        # indices into original indice for the cache
        retrieved_indices = torch.unsqueeze(retrieved_indices, dim=1).detach()
        # change from 1D to 2D: [batch_size] --> [batch_size x 1]
    else:
        retrieved_indices = retrieval_fn(scores)
        retrieved_indices = retrieved_indices.detach()
    retrieved_data = {
        k: util.gather_nd(v.detach().to(device), retrieved_indices).detach()
        for k, v in all_data.items()
    }
    # retrieved_data contains Tensors with values (specified in specs without embedding == data_keys)
    # the tensor data is the data for the value which index was retrieved via the retrieval
    # function (the gumbel max retrieval fn)
    # each row in the tensor is for one query, since each query got one index retrieved

    if top_k and return_topk == 1:
        # more efficient than computing top-1 element twice
        # this way the top element is samlpled by gumbel-max
        retrieved_topk_features = [retrieved_data]

    retrieved_cache_embeddings = util.gather_nd(
        all_embeddings, retrieved_indices
    ).detach()
    # retrieved_cache_embeddings: same as retrieved_data but not as a dict. Contains the embeddings of
    # the documents which scores had the highest gumbel max (with gaumbel value and inv_temp)
    del all_data
    return _RetrievalReturn(
        retrieved_data,
        scores,
        retrieved_indices,
        retrieved_cache_embeddings,
        retrieved_topk_features,
    )


def _get_data_sorce_start_position_and_cache_sizes(
    cache, embedding_key, sorted_data_sources
):
    """Gets the first index and size per data sources in the concatenated data."""
    curr_position = torch.tensor(0, dtype=torch.int64).detach()
    start_positions = {}
    cache_sizes = {}
    for data_source in sorted_data_sources:
        start_positions[data_source] = curr_position
        cache_sizes[data_source] = cache[data_source].data[embedding_key].size()[0]
        curr_position = curr_position + cache_sizes[data_source]
    return start_positions, cache_sizes


def _get_retrieved_embedding_updates(
    cache, embedding_key, sorted_data_sources, retrieved_indices, retrieved_embeddings
):
    """Gets the updates for the retrieved data."""
    updated_item_indices = {}
    updated_item_data = {}
    updated_item_mask = {}

    start_positions, cache_sizes = _get_data_sorce_start_position_and_cache_sizes(
        cache, embedding_key, sorted_data_sources
    )

    for data_source in sorted_data_sources:
        updated_item_indices[data_source] = (
            retrieved_indices - start_positions[data_source]
        )
        updated_item_data[data_source] = {embedding_key: retrieved_embeddings}
        updated_item_mask[data_source] = (
            retrieved_indices >= start_positions[data_source]
        ) & (
            retrieved_indices < start_positions[data_source] + cache_sizes[data_source]
        )
        updated_item_indices[data_source] = torch.squeeze(
            updated_item_indices[data_source], dim=1
        ).detach()
        updated_item_mask[data_source] = torch.squeeze(
            updated_item_mask[data_source], dim=1
        ).detach()

    # updated_item_data: dict with new embeddings as data
    # updated_item_indices: indices of new data on each cache
    # updated_item_mask: contains information whether the new data should be updated
    # into the respectiv cache
    del start_positions
    del cache_sizes
    return updated_item_data, updated_item_indices, updated_item_mask


def _get_staleness(cache_embeddings, updated_embeddings):
    error = cache_embeddings - updated_embeddings
    mse = torch.sum(error ** 2, dim=1).detach()
    normalized_mse = mse / torch.sum(updated_embeddings ** 2, dim=1).detach()
    return normalized_mse.detach()


_LossCalculationReturn = collections.namedtuple(
    "_LossCalculationReturn",
    [
        "training_loss",
        "interpretable_loss",
        "staleness",
        "retrieval_return",
        "retrieved_negative_embeddings",
        "retrieved_topk_features",
    ],
)


class AbstractCacheClassificationLoss(CacheLoss, metaclass=abc.ABCMeta):
    """Abstract method for cache classification losses.

    Inherit from this object and override `_retrieve_from_cache` and
    `_score_documents` to implement a cache classification loss based on the
    specified retrieval and scoring approaches.
    """

    @abc.abstractmethod
    def _retrieve_from_cache(self, query_embeddings, cache, return_topk=0):
        pass

    @abc.abstractmethod
    def _score_documents(self, query_embeddings, doc_embeddings):
        pass

    def _calculate_training_loss_and_summaries(
        self,
        doc_network,
        query_embeddings,
        pos_doc_embeddings,
        cache,
        return_topk=0,
        reducer=torch.mean,
    ):
        """Calculates the cache classification loss and associated summaries."""
        positive_scores = self._score_documents(query_embeddings, pos_doc_embeddings)
        retrieval_return = self._retrieve_from_cache(
            query_embeddings, cache, return_topk
        )

        retrieved_negative_embeddings = doc_network(retrieval_return.retrieved_data)

        retrieved_negative_scores = self._score_documents(
            query_embeddings, retrieved_negative_embeddings
        )
        # retrieved_negative_scores: [batch_size] <-- query_embedding * transposed(retrieved scores),
        # one score for each query b'cos only one document retrieved per query

        cache_and_pos_scores = torch.concat(
            [torch.unsqueeze(positive_scores, dim=1), retrieval_return.scores,], dim=1,
        )
        # score of query and positive document embedding from current batch are concatenated with the
        # approximated top_k scores from the cache
        # cache_and_pos_scores: [batch_size x top_k + 1]

        prob_pos = F.softmax(cache_and_pos_scores, dim=1)[:, 0]
        prob_pos = prob_pos.detach()
        training_loss = (1.0 - prob_pos) * (retrieved_negative_scores - positive_scores)
        interpretable_loss = -torch.log(prob_pos).detach()
        staleness = _get_staleness(
            retrieval_return.retrieved_cache_embeddings, retrieved_negative_embeddings
        ).detach()
        if reducer is not None:
            training_loss = reducer(training_loss)
            interpretable_loss = reducer(interpretable_loss)
            staleness = reducer(staleness)
        retrieved_negative_embeddings = retrieved_negative_embeddings.detach()
        return _LossCalculationReturn(
            training_loss=training_loss,
            interpretable_loss=interpretable_loss,
            staleness=staleness,
            retrieval_return=retrieval_return,
            retrieved_negative_embeddings=retrieved_negative_embeddings,
            retrieved_topk_features=retrieval_return.retrieved_cache_topk_features,
        )


class CacheClassificationLoss(AbstractCacheClassificationLoss):
    """Implements an efficient way to train with a cache classification loss.

    The cache classification loss is the negative log probability of the positive
    document when the distribution is the softmax of all documents. This object
    allows calculating:
      (1) An efficient stochastic loss function whose gradient is approximately
          the same as the cache classification loss in expectation. This gradient
          can be calculated by feeding only O(batch_size) documents through the
          document network, rather than O(cache_size) for the standard
          implementation.
      (2) An approximation of the value cache classification loss using the cached
          embeddings. The loss described above is not interpretable. This loss is
          a direct approximation of the cache classification loss, however we
          cannot calculate a gradient of this loss.

    Calling the CacheClassificationLoss return a CacheLossReturn object, which
    has the following fields:
      training_loss: Use this to calculate gradients.
      interpretable_loss: An interpretable number for the CacheClassificationLoss
          to use as a Tensorboard summary.
      updated_item_data, updated_item_indices, updated_item_mask: Use these in
          the negative cache updates. These describe the cache elements that were
          retrieved and current embedding calculated.
      staleness: This is the square error between the retrieved cache embeddings
          and the retrieved embeddings as defined by the current state of the
          model. Create a summary of this value as a proxy for the error due to
          cache staleness.
    """

    def __init__(
        self,
        embedding_key,
        data_keys,
        score_transform=None,
        top_k=None,
        reducer=torch.mean,
    ):
        """Initializes the CacheClassificationLoss object.

        Args:
          embedding_key: The key containing the embedding in the cache.
          data_keys: The keys containing the document data in the cache.
          score_transform: Scores are transformed by this function before use.
            Specifically we have scores(i, j) = score_transform(dot(query_embed_i,
            doc_embed_j))
          top_k: If set, the top k scoring negative elements will be mined and the
            rest of the elements masked before calculating the loss.
          reducer: Function that reduces the losses to a single scaler. If None,
            then the elementwise losses are returned.
        """
        self.embedding_key = embedding_key
        self.data_keys = data_keys
        self.score_transform = score_transform
        self.top_k = top_k
        self.reducer = reducer
        self._retrieval_fn = retrieval_fns.GumbelMaxRetrievalFn()

    def _retrieve_from_cache(self, query_embeddings, cache, return_topk=0):
        sorted_data_sources = sorted(cache.keys())
        return _retrieve_from_caches(
            query_embeddings,
            cache,
            self._retrieval_fn,
            self.embedding_key,
            self.data_keys,
            sorted_data_sources,
            return_topk,
            self.score_transform,
            self.top_k,
        )

    def _score_documents(self, query_embeddings, doc_embeddings):
        return _score_documents(
            query_embeddings, doc_embeddings, score_transform=self.score_transform
        )

    def __call__(
        self, doc_network, query_embeddings, pos_doc_embeddings, cache, return_topk=0
    ):
        """Calculates the cache classification losses.

        Args:
          doc_network: The network that embeds the document data.
          query_embeddings: Embeddings for the queries.
          pos_doc_embeddings: Embeddings for the documents that are positive for the
            given queries.
          cache: The cache of document data and embeddings.

        Returns:
          A CacheLossReturn object with the training loss, interpretable loss, and
          data needed to update the cache element embeddings that were retrieved and
          recalculated.
        """
        loss_calculation_return = self._calculate_training_loss_and_summaries(
            doc_network,
            query_embeddings,
            pos_doc_embeddings,
            cache,
            return_topk,
            self.reducer,
        )

        training_loss = loss_calculation_return.training_loss
        interpretable_loss = loss_calculation_return.interpretable_loss
        staleness = loss_calculation_return.staleness
        retrieval_return = loss_calculation_return.retrieval_return
        # retrieved_negative_embeddings: new computed embeddings using the document_network and
        # one gumbel max retrieved item from the cache, per query.
        # Size = [batch_size x doc_network embedding size == CLS token embedding for BERT == 768] --> [batch_size x 768]
        retrieved_negative_embeddings = (
            loss_calculation_return.retrieved_negative_embeddings.detach()
        )
        topk_features = retrieval_return.retrieved_cache_topk_features

        sorted_data_sources = sorted(cache.keys())

        (
            updated_item_data,
            updated_item_indices,
            updated_item_mask,
        ) = _get_retrieved_embedding_updates(
            cache,
            self.embedding_key,
            sorted_data_sources,
            retrieval_return.retrieved_indices,
            retrieved_negative_embeddings,
        )

        return CacheLossReturn(
            training_loss=training_loss,
            interpretable_loss=interpretable_loss,
            updated_item_data=updated_item_data,
            updated_item_indices=updated_item_indices,
            updated_item_mask=updated_item_mask,
            staleness=staleness,
            topk_features=topk_features,
        )


def _get_local_elements_global_data(all_elements_local_data, num_replicas):
    all_elements_local_data = torch.unsqueeze(all_elements_local_data, dim=1).detach()
    list_of_tensors = list(
        torch.tensor_split(input=all_elements_local_data, sections=num_replicas, dim=0)
    )
    gathered_tensor_list = [
        torch.empty_like(list_of_tensors[0]).detach() for _ in range(num_replicas)
    ]
    dist.all_to_all(
        output_tensor_list=gathered_tensor_list, input_tensor_list=list_of_tensors
    )
    concat = torch.concat(gathered_tensor_list, dim=1).detach()
    del list_of_tensors
    del gathered_tensor_list
    return concat


class DistributedCacheClassificationLoss(AbstractCacheClassificationLoss):
    """Implements a cache classification loss with a sharded cache.

    This object implements a cache classification loss when the cache is sharded
    onto multiple replicas. This code calculates the loss treating the sharded
    cache as one unit, so all queries are affected by all cache elements in every
    replica.

    Currently, the updated_item_* fields (i.e., the embedding updates for items
    already in the cache) in the CacheLossReturn are empty. This does not affect
    new items introduced to the cache.
    """

    def __init__(
        self,
        embedding_key,
        data_keys,
        score_transform=None,
        top_k=None,
        reducer=torch.mean,
    ):
        self.embedding_key = embedding_key
        self.data_keys = data_keys
        self.score_transform = score_transform
        self.top_k = top_k
        self.reducer = reducer
        self._retrieval_fn = retrieval_fns.GumbelMaxRetrievalFn()

    def _score_documents(self, query_embeddings, doc_embeddings):
        return _score_documents(
            query_embeddings, doc_embeddings, score_transform=self.score_transform
        )

    def _retrieve_from_cache(self, query_embeddings, cache, return_topk=0):
        sorted_data_sources = sorted(cache.keys())
        all_query_embeddings = util.cross_replica_concat(query_embeddings)
        num_replicas = dist.get_world_size()
        # Performs approximate top k across replicas.
        if self.top_k:
            top_k_per_replica = self.top_k // num_replicas
        else:
            top_k_per_replica = self.top_k
        retrieval_return = _retrieve_from_caches(
            all_query_embeddings,
            cache,
            self._retrieval_fn,
            self.embedding_key,
            self.data_keys,
            sorted_data_sources,
            return_topk,
            self.score_transform,
            top_k_per_replica,
        )

        # We transfer all queries to all replica and retrieve from every shard.
        all_queries_local_weight = torch.logsumexp(
            retrieval_return.scores, dim=1
        ).detach()
        local_queries_global_weights = _get_local_elements_global_data(
            all_queries_local_weight, num_replicas
        )
        local_queries_all_retrieved_data = {}
        for key in retrieval_return.retrieved_data:
            local_queries_all_retrieved_data[key] = _get_local_elements_global_data(
                retrieval_return.retrieved_data[key], num_replicas
            )
        local_queries_all_retrieved_embeddings = _get_local_elements_global_data(
            retrieval_return.retrieved_cache_embeddings, num_replicas
        )
        # We then sample a shard index proportional to its total weight.
        # This allows us to do Gumbel-Max sampling without modifying APIs.
        selected_replica = self._retrieval_fn(local_queries_global_weights)
        selected_replica = selected_replica.detach()
        num_elements = selected_replica.size()[0]
        with torch.cuda.device(dist.get_rank()):
            batch_indices = torch.arange(end=num_elements).detach().cuda()
            batch_indices = batch_indices.type(torch.int64)
            batch_indices = torch.unsqueeze(batch_indices, dim=1).detach()
            selected_replica_with_batch = (
                torch.concat([batch_indices, selected_replica], dim=1).detach().cuda()
            )

        # We retrieved topk / num_replica features per query per replica. We have to concatenate the
        # retrieved features for the respective queries.
        if return_topk:
            retrieved_topk_features_global = []
            for idx in range(
                0, len(retrieval_return.retrieved_cache_topk_features), num_replicas
            ):
                new_dict = {}
                topk_per_query = retrieval_return.retrieved_cache_topk_features[
                    idx : idx + num_replicas
                ]
                for d in topk_per_query:
                    for k, v in d.items():
                        if k in new_dict:
                            new_dict[k] = torch.vstack((new_dict[k], v))
                        else:
                            new_dict[k] = v
                retrieved_topk_features_global.append(new_dict)
        else:
            retrieved_topk_features_global = None

        retrieved_data = {
            k: util.gather_nd(v, selected_replica_with_batch).detach()
            for k, v in local_queries_all_retrieved_data.items()
        }
        retrieved_cache_embeddings = util.gather_nd(
            local_queries_all_retrieved_embeddings, selected_replica_with_batch
        ).detach()
        del retrieval_return
        del all_queries_local_weight
        del local_queries_all_retrieved_data
        del local_queries_all_retrieved_embeddings
        del selected_replica_with_batch
        return _RetrievalReturn(
            retrieved_data=retrieved_data,
            scores=local_queries_global_weights,
            retrieved_indices=None,
            retrieved_cache_embeddings=retrieved_cache_embeddings,
            retrieved_cache_topk_features=retrieved_topk_features_global,
        )

    def __call__(
        self, doc_network, query_embeddings, pos_doc_embeddings, cache, return_topk=0
    ):
        loss_calculation_return = self._calculate_training_loss_and_summaries(
            doc_network,
            query_embeddings,
            pos_doc_embeddings,
            cache,
            return_topk,
            self.reducer,
        )
        training_loss = loss_calculation_return.training_loss
        interpretable_loss = loss_calculation_return.interpretable_loss
        staleness = loss_calculation_return.staleness
        topk_features = loss_calculation_return.retrieved_topk_features
        del loss_calculation_return
        return CacheLossReturn(
            training_loss=training_loss,
            interpretable_loss=interpretable_loss,
            updated_item_data={k: None for k in cache},
            updated_item_indices={k: None for k in cache},
            updated_item_mask={k: None for k in cache},
            staleness=staleness,
            topk_features=topk_features,
        )

