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

"""Tests for negative_cache.handler."""

import torch
import unittest
import pytest
from negative_cache import handlers
from negative_cache import losses
from negative_cache import negative_cache
from negative_cache.negative_cache import FixedLenFeature


class StubCacheLoss(object):
    def __init__(self, updated_item_data, updated_item_indices, updated_item_mask):
        self.updated_item_data = updated_item_data
        self.updated_item_indices = updated_item_indices
        self.updated_item_mask = updated_item_mask

    def __call__(self, doc_network, query_embeddings, pos_doc_embeddings, cache):
        return losses.CacheLossReturn(
            training_loss=0.0,
            interpretable_loss=0.0,
            updated_item_data=self.updated_item_data,
            updated_item_indices=self.updated_item_indices,
            updated_item_mask=self.updated_item_mask,
            staleness=0.0,
        )


class HandlerTest(unittest.TestCase):
    def test_initialize_cache(self):
        specs = {
            "data": FixedLenFeature(shape=[2], dtype=torch.int32),
            "embedding": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs, cache_size=4)
        handler = handlers.CacheLossHandler(
            cache_manager,
            StubCacheLoss(None, None, None),
            embedding_key="embedding",
            data_keys=("data",),
        )
        self.assertSetEqual({"data", "embedding"}, set(handler.cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.zeros(size=[4, 2], dtype=torch.int32), handler.cache.data["data"]
            )
        )
        self.assertTrue(
            torch.equal(
                torch.zeros(size=[4, 3], dtype=torch.float32),
                handler.cache.data["embedding"],
            )
        )
        self.assertTrue(
            torch.equal(torch.zeros(size=[4], dtype=torch.int32), handler.cache.age)
        )

    @pytest.mark.xfail
    def test_check_cache_after_update(self):
        """
        NOTE: The original implementation of negative_cache used tf.math.top_k which returned the lowest
        indeces for duplicate values. The torch implementation of topk returns any indices for duplicates.
        There is no specific rule for indices of duplicates. Because of this the test cases needed to be
        changed to match the torch.topk behavior. I tried to make this as reproducible as possible by
        seeding torch. Test was written with: torch==1.10.1 and python 3.9.7
        """
        torch.manual_seed(42)

        specs = {
            "data": FixedLenFeature(shape=[2], dtype=torch.int32),
            "embedding": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs, cache_size=4)
        cache_loss = StubCacheLoss(
            updated_item_data={"cache": {"embedding": torch.tensor([[1.0, 1.0, 1.0]])}},
            updated_item_indices={"cache": torch.tensor([0])},
            updated_item_mask={"cache": torch.tensor([True])},
        )
        handler = handlers.CacheLossHandler(
            cache_manager, cache_loss, embedding_key="embedding", data_keys=("data",)
        )
        loss_actual = handler.update_cache_and_compute_loss(
            item_network=None,
            query_embeddings=None,
            pos_item_embeddings=torch.tensor([[2.0, 2.0, 2.0]]),
            features={"data": torch.tensor([[2, 2]])},
        )
        self.assertSetEqual({"data", "embedding"}, set(handler.cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[0, 0], [0, 0], [2, 2], [0, 0]], dtype=torch.int32),
                handler.cache.data["data"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [2.0, 2.0, 2.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                handler.cache.data["embedding"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor([0, 1, 0, 1], dtype=torch.int32), handler.cache.age
            )
        )
        self.assertEqual(0.0, loss_actual)


if __name__ == "__main__":
    unittest.main()
