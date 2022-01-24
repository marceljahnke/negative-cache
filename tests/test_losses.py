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

# Lint as: python3
"""Tests for negative_cache.losses."""

import torch
import unittest
from negative_cache import losses
from negative_cache import negative_cache


class LossesTest(unittest.TestCase):

    # tensorflow assertAllEqual relative and absolute tolerances
    rtol = 1e-06
    atol = 1e-06

    def test_cache_classification_loss_interpretable_loss(self):
        cached_embeddings = torch.tensor([[1.0], [-1.0]])
        cached_data = torch.tensor([[1.0], [-1.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[-1.0], [1.0], [3.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0], [1.0]])

        loss_fn = losses.CacheClassificationLoss("embeddings", ["data"], reducer=None)
        interpretable_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).interpretable_loss
        interpretable_loss_expected = torch.tensor([3.169846, 0.349012, 0.694385])
        self.assertTrue(
            torch.allclose(
                interpretable_loss_expected,
                interpretable_loss,
                rtol=self.rtol,
                atol=self.atol,
            )
        )

        loss_fn.reducer = torch.mean
        interpretable_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).interpretable_loss
        interpretable_loss_expected = torch.tensor(
            (3.169846 + 0.349012 + 0.694385) / 3.0
        )
        self.assertTrue(
            torch.allclose(
                interpretable_loss_expected,
                interpretable_loss,
                rtol=self.rtol,
                atol=self.atol,
            )
        )

    def test_cache_classification_loss_training_loss(self):
        cached_embeddings = torch.tensor([[1.0], [-1.0]])
        cached_data = torch.tensor([[1.0], [-1.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[-1.0], [1.0], [3.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0], [1.0]])

        # pylint: disable=g-long-lambda
        def retrieval_fn(scores):
            return torch.tensor([[0], [1], [0]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        loss_fn = losses.CacheClassificationLoss("embeddings", ["data"], reducer=None)
        loss_fn._retrieval_fn = retrieval_fn
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss
        prob_pos = torch.tensor([0.0420101, 0.705385, 0.499381])
        score_differences = torch.tensor([1.0, -3.0, 0.0])
        training_loss_expected = (1.0 - prob_pos) * score_differences
        torch.set_printoptions(precision=7)
        self.assertTrue(
            torch.allclose(
                training_loss_expected, training_loss, rtol=self.rtol, atol=self.atol
            )
        )

        loss_fn.reducer = torch.mean
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss
        training_loss_expected = torch.mean((1.0 - prob_pos) * score_differences)
        self.assertTrue(
            torch.allclose(
                training_loss_expected, training_loss, rtol=self.rtol, atol=self.atol
            )
        )

    def test_cache_classification_loss_training_loss_with_score_transform(self):
        cached_embeddings = 2.0 * torch.tensor([[1.0], [-1.0]])
        cached_data = 2.0 * torch.tensor([[1.0], [-1.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }

        def doc_network(data):
            return data["data"]

        query_embedding = 2.0 * torch.tensor([[-1.0], [1.0], [3.0]])
        pos_doc_embedding = 2.0 * torch.tensor([[2.0], [2.0], [1.0]])

        # pylint: disable=g-long-lambda
        def retrieval_fn(scores):
            return torch.tensor([[0], [1], [0]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        def score_transform(scores):
            return 0.25 * scores

        loss_fn = losses.CacheClassificationLoss(
            "embeddings", ["data"], score_transform=score_transform, reducer=None
        )
        loss_fn._retrieval_fn = retrieval_fn
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss
        prob_pos = torch.tensor([0.0420101, 0.705385, 0.499381])
        score_differences = torch.tensor([1.0, -3.0, 0.0])
        training_loss_expected = (1.0 - prob_pos) * score_differences
        self.assertTrue(
            torch.allclose(
                training_loss_expected, training_loss, rtol=self.rtol, atol=self.atol
            )
        )

        loss_fn.reducer = torch.mean
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss
        training_loss_expected = torch.mean((1.0 - prob_pos) * score_differences)
        self.assertTrue(
            torch.allclose(
                training_loss_expected, training_loss, rtol=self.rtol, atol=self.atol
            )
        )

    def test_cache_classification_loss_training_loss_gradient(self):
        cached_embeddings = torch.tensor([[1.0], [-1.0]])
        cached_data = torch.tensor([[1.0], [-1.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }
        query_model = torch.tensor(1.0, requires_grad=True)
        doc_model = torch.tensor(1.0, requires_grad=True)

        def doc_network(data):
            return data["data"] * doc_model

        query_embedding = torch.tensor([[-1.0], [1.0], [3.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0], [1.0]])

        # pylint: disable=g-long-lambda
        def retrieval_fn(scores):
            return torch.tensor([[0], [1], [0]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        loss_fn = losses.CacheClassificationLoss("embeddings", ["data"])
        loss_fn._retrieval_fn = retrieval_fn

        query_embedding = query_model * query_embedding
        pos_doc_embedding = doc_model * pos_doc_embedding
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss

        training_loss.backward()

        # gradient should have same floating point precision for comparison
        torch.set_printoptions(precision=9)  # default=4
        gradient = torch.tensor([query_model.grad, doc_model.grad])
        torch.set_printoptions(precision=4)

        gradient_expected = torch.tensor([0.024715006, 0.024715006])
        self.assertTrue(
            torch.allclose(gradient_expected, gradient, rtol=self.rtol, atol=self.atol)
        )

    def test_cache_classification_loss_training_loss_with_multi_cache(self):
        cached_embeddings_1 = torch.tensor([[1.0]])
        cached_embeddings_2 = torch.tensor([[-1.0]])
        cached_data_1 = torch.tensor([[1.0]])
        cached_data_2 = torch.tensor([[-1.0]])
        cache = {
            "cache1": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings_1, "data": cached_data_1}, age=[0]
            ),
            "cache2": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings_2, "data": cached_data_2}, age=[0]
            ),
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[-1.0], [1.0], [3.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0], [1.0]])

        # pylint: disable=g-long-lambda
        def retrieval_fn(scores):
            return torch.tensor([[0], [1], [0]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        loss_fn = losses.CacheClassificationLoss("embeddings", ["data"], reducer=None)
        loss_fn._retrieval_fn = retrieval_fn
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss
        prob_pos = torch.tensor([0.0420101, 0.705385, 0.499381])
        score_differences = torch.tensor([1.0, -3.0, 0.0])
        training_loss_expected = (1.0 - prob_pos) * score_differences
        self.assertTrue(
            torch.allclose(
                training_loss_expected, training_loss, rtol=self.rtol, atol=self.atol
            )
        )

        loss_fn.reducer = torch.mean
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss
        training_loss_expected = torch.mean((1.0 - prob_pos) * score_differences)
        self.assertTrue(
            torch.allclose(
                training_loss_expected, training_loss, rtol=self.rtol, atol=self.atol
            )
        )

    def test_cache_classification_loss_refreshed_embeddings(self):
        cached_embeddings_1 = torch.tensor([[1.0], [2.0], [3.0]])
        cached_embeddings_2 = torch.tensor([[-1.0], [-2.0]])
        cached_data_1 = torch.tensor([[10.0], [20.0], [30.0]])
        cached_data_2 = torch.tensor([[-10.0], [-20.0]])
        cache = {
            "cache1": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings_1, "data": cached_data_1},
                age=[0, 0, 0],
            ),
            "cache2": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings_2, "data": cached_data_2},
                age=[0, 0],
            ),
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[0.0], [0.0], [0.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0], [1.0]])

        # pylint: disable=g-long-lambda
        def retrieval_fn(scores):
            return torch.tensor([[0], [1], [3]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        loss_fn = losses.CacheClassificationLoss("embeddings", ["data"])
        loss_fn._retrieval_fn = retrieval_fn
        cache_loss_return = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        )
        self.assertTrue(
            torch.equal(
                cache_loss_return.updated_item_mask["cache1"],
                torch.tensor([True, True, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                cache_loss_return.updated_item_mask["cache2"],
                torch.tensor([False, False, True]),
            )
        )
        self.assertTrue(
            torch.equal(
                cache_loss_return.updated_item_data["cache1"]["embeddings"][0:2],
                torch.tensor([[10.0], [20.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                cache_loss_return.updated_item_data["cache2"]["embeddings"][2],
                torch.tensor([-10.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                cache_loss_return.updated_item_indices["cache1"][0:2],
                torch.tensor([0, 1]),
            )
        )
        self.assertEqual(cache_loss_return.updated_item_indices["cache2"][2], 0)

    def test_cache_classification_loss_staleness(self):
        cached_embeddings = torch.tensor([[1.0], [-1.0]])
        cached_data = torch.tensor([[2.0], [-3.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[0.0], [0.0], [0.0]])
        pos_doc_embedding = torch.tensor([[0.0], [0.0], [0.0]])

        # pylint: disable=g-long-lambda
        def retrieval_fn(scores):
            return torch.tensor([[0], [1], [0]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        loss_fn = losses.CacheClassificationLoss("embeddings", ["data"])
        loss_fn._retrieval_fn = retrieval_fn
        staleness = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).staleness
        staleness_expected = torch.tensor(0.31481481481)
        self.assertTrue(
            torch.allclose(
                staleness_expected, staleness, rtol=self.rtol, atol=self.atol
            )
        )

        loss_fn.reducer = None
        staleness = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).staleness
        staleness_expected = torch.tensor([1.0 / 4.0, 4.0 / 9.0, 1.0 / 4.0])
        self.assertTrue(
            torch.allclose(
                staleness_expected, staleness, rtol=self.rtol, atol=self.atol
            )
        )

    def test_cache_classification_loss_interpretable_loss_with_top_k(self):
        cached_embeddings = torch.tensor([[1.0], [-1.0], [3.0], [2.0]])
        cached_data = torch.tensor([[1.0], [-1.0], [3.0], [2.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[-1.0], [1.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0]])

        loss_fn = losses.CacheClassificationLoss(
            "embeddings", ["data"], reducer=None, top_k=2
        )
        interpretable_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).interpretable_loss
        interpretable_loss_expected = torch.tensor([3.0949229, 1.407605964])
        self.assertTrue(
            torch.allclose(
                interpretable_loss_expected,
                interpretable_loss,
                rtol=self.rtol,
                atol=self.atol,
            )
        )

    def test_cache_classification_loss_training_loss_with_top_k(self):
        cached_embeddings = torch.tensor([[1.0], [-1.0], [3.0], [2.0]])
        cached_data = torch.tensor([[1.0], [-1.0], [3.0], [2.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[-1.0], [1.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0]])

        loss_fn = losses.CacheClassificationLoss(
            "embeddings", ["data"], reducer=None, top_k=2
        )

        # pylint: disable=g-long-lambda
        def retrieval_fn(scores):
            return torch.tensor([[0], [0]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        loss_fn._retrieval_fn = retrieval_fn
        training_loss = loss_fn(
            doc_network, query_embedding, pos_doc_embedding, cache
        ).training_loss
        prob_pos = torch.tensor([0.045278503, 0.2447284712])
        score_differences = torch.tensor([3.0, -1.0])
        training_loss_expected = (1.0 - prob_pos) * score_differences
        self.assertTrue(
            torch.allclose(
                training_loss_expected, training_loss, rtol=self.rtol, atol=self.atol
            )
        )

    def test_cache_classification_loss_refreshed_embeddings_with_top_k(self):
        cached_embeddings = torch.tensor([[1.0], [-1.0], [3.0], [2.0]])
        cached_data = torch.tensor([[1.0], [-1.0], [3.0], [2.0]])
        cache = {
            "cache": negative_cache.NegativeCache(
                data={"embeddings": cached_embeddings, "data": cached_data}, age=[0, 0]
            )
        }

        def doc_network(data):
            return data["data"]

        query_embedding = torch.tensor([[-1.0], [1.0]])
        pos_doc_embedding = torch.tensor([[2.0], [2.0]])

        loss_fn = losses.CacheClassificationLoss("embeddings", ["data"], top_k=2)
        # pylint: disable=g-long-lambda

        def retrieval_fn(scores):
            return torch.tensor([[0], [0]], dtype=torch.int64)

        # pylint: enable=g-long-lambda
        loss_fn._retrieval_fn = retrieval_fn
        loss_fn_return = loss_fn(doc_network, query_embedding, pos_doc_embedding, cache)
        updated_item_data = loss_fn_return.updated_item_data
        updated_item_indices = loss_fn_return.updated_item_indices
        updated_item_mask = loss_fn_return.updated_item_mask
        self.assertTrue(
            torch.allclose(
                torch.tensor([[-1.0], [1.0]]),
                updated_item_data["cache"]["embeddings"],
                rtol=self.rtol,
                atol=self.atol,
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([1, 0]), updated_item_indices["cache"])
        )
        self.assertTrue(
            torch.equal(torch.tensor([True, True]), updated_item_mask["cache"])
        )


if __name__ == "__main__":
    unittest.main()
