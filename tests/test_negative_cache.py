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
"""Tests for negative_cache.negative_cache."""

import torch
import unittest
import pytest
from negative_cache import negative_cache
from negative_cache.negative_cache import FixedLenFeature


class NegativeCacheTest(unittest.TestCase):
    def test_init_cache(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
            "3": FixedLenFeature(shape=[3, 2], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=6)
        cache = cache_manager.init_cache()
        self.assertEqual({"1", "2", "3"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(torch.zeros([6, 2], dtype=torch.int32), cache.data["1"])
        )
        self.assertTrue(
            torch.equal(torch.zeros([6, 3], dtype=torch.float32), cache.data["2"])
        )
        self.assertTrue(
            torch.equal(torch.zeros([6, 3, 2], dtype=torch.float32), cache.data["3"])
        )
        self.assertTrue(torch.equal(torch.zeros([6], dtype=torch.int32), cache.age))

    @pytest.mark.xfail
    def test_update_cache(self):
        """
        NOTE: The original implementation of negative_cache used tf.math.top_k which returned the lowest
        indeces for duplicate values. The torch implementation of topk returns any indices for duplicates.
        There is no specific rule for indices of duplicates. Because of this the test cases needed to be
        changed to match the torch.topk behavior. I tried to make this as reproducible as possible by
        seeding torch. Test was written with: torch==1.10.1 and python 3.9.7
        """
        torch.manual_seed(42)

        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = cache_manager.init_cache()
        updates = {
            "1": torch.ones(size=[2, 2], dtype=torch.int32),
            "2": torch.ones(size=[2, 3], dtype=torch.float32),
        }
        cache = cache_manager.update_cache(cache, new_items=updates)

        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        print(cache.data["1"])
        self.assertTrue(
            torch.equal(
                torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([1, 1, 0, 0], dtype=torch.int32), cache.age)
        )

        updates = {
            "1": 2 * torch.ones(size=[2, 2], dtype=torch.int32),
            "2": 2.0 * torch.ones(size=[2, 3], dtype=torch.float32),
        }
        cache = cache_manager.update_cache(cache, new_items=updates)
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[2, 2], [2, 2], [1, 1], [1, 1]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )

        updates = {
            "1": 3 * torch.ones(size=[2, 2], dtype=torch.int32),
            "2": 3.0 * torch.ones(size=[2, 3], dtype=torch.float32),
        }
        cache = cache_manager.update_cache(cache, new_items=updates)
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[2, 2], [2, 2], [3, 3], [3, 3]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0],
                    ],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )

    def test_update_cache_with_non_multiple_cache_size(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=3)
        cache = cache_manager.init_cache()
        updates = {
            "1": torch.ones(size=[2, 2], dtype=torch.int32),
            "2": torch.ones(size=[2, 3], dtype=torch.float32),
        }
        cache = cache_manager.update_cache(cache, new_items=updates)
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[1, 1], [1, 1], [0, 0]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([0, 0, 1], dtype=torch.int32), cache.age)
        )
        updates = {
            "1": 2 * torch.ones(size=[2, 2], dtype=torch.int32),
            "2": 2.0 * torch.ones(size=[2, 3], dtype=torch.float32),
        }
        cache = cache_manager.update_cache(cache, new_items=updates)
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[2, 2], [1, 1], [2, 2]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([0, 1, 0], dtype=torch.int32), cache.age)
        )

    @pytest.mark.xfail
    def test_update_caches_with_function(self):
        """
        NOTE: The original implementation of negative_cache used tf.math.top_k which returned the lowest
        indeces for duplicate values. The torch implementation of topk returns any indices for duplicates.
        There is no specific rule for indices of duplicates. Because of this the test cases needed to be
        changed to match the torch.topk behavior. I tried to make this as reproducible as possible by
        seeding torch. Test was written with: torch==1.10.1 and python 3.9.7
        """
        torch.manual_seed(42)

        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        init_cache_fn = cache_manager.init_cache
        cache = init_cache_fn()
        updates = {
            "1": torch.ones(size=[2, 2], dtype=torch.int32),
            "2": torch.ones(size=[2, 3], dtype=torch.float32),
        }

        update_cache_fn = cache_manager.update_cache
        cache = update_cache_fn(cache, new_items=updates)
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )

    def test_raises_value_error_if_different_update_sizes(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        init_cache_fn = cache_manager.init_cache
        cache = init_cache_fn()
        updates = {
            "1": torch.ones(size=[2, 2], dtype=torch.int32),
            "2": torch.ones(size=[1, 3], dtype=torch.float32),
        }
        update_cache_fn = cache_manager.update_cache
        with self.assertRaises(IndexError):
            cache = update_cache_fn(cache, new_items=updates)

    def test_update_cache_with_existing_items(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = cache_manager.init_cache()
        updated_item_indices = torch.tensor([1, 3], dtype=torch.int32)
        updated_item_data = {
            "1": torch.ones(size=[2, 2], dtype=torch.int32),
            "2": torch.ones(size=[2, 3], dtype=torch.float32),
        }
        cache = cache_manager.update_cache(
            cache,
            updated_item_data=updated_item_data,
            updated_item_indices=updated_item_indices,
        )
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([1, 0, 1, 0], dtype=torch.int32), cache.age)
        )

    def test_partial_update_cache_with_existing_items(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = cache_manager.init_cache()
        updated_item_indices = torch.tensor([1, 3], dtype=torch.int32)
        updated_item_data = {
            "1": torch.ones(size=[2, 2], dtype=torch.int32),
        }
        cache = cache_manager.update_cache(
            cache,
            updated_item_data=updated_item_data,
            updated_item_indices=updated_item_indices,
        )
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(torch.zeros([4, 3], dtype=torch.float32), cache.data["2"])
        )
        self.assertTrue(
            torch.equal(torch.tensor([1, 0, 1, 0], dtype=torch.int32), cache.age)
        )

    def test_update_cache_with_new_items_and_existing_items(self):
        specs = {
            "1": FixedLenFeature(shape=[1], dtype=torch.int32),
            "2": FixedLenFeature(shape=[1], dtype=torch.int32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=2)
        data = {
            "1": torch.tensor([[0], [0], [3]], dtype=torch.int32),
            "2": torch.tensor([[1], [2], [4]], dtype=torch.int32),
        }
        age = torch.tensor([2, 1, 0], dtype=torch.int32)
        cache = negative_cache.NegativeCache(data=data, age=age)
        updated_item_indices = torch.tensor([0], dtype=torch.int32)
        updated_item_data = {
            "1": torch.tensor([[10]], dtype=torch.int32),
        }
        new_items = {
            "1": torch.tensor([[11]], dtype=torch.int32),
            "2": torch.tensor([[12]], dtype=torch.int32),
        }
        cache = cache_manager.update_cache(
            cache,
            new_items=new_items,
            updated_item_data=updated_item_data,
            updated_item_indices=updated_item_indices,
        )
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[10], [11], [3]], dtype=torch.int32), cache.data["1"]
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor([[1], [12], [4]], dtype=torch.int32), cache.data["2"]
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([0, 0, 1], dtype=torch.int32), cache.age)
        )

    def test_raises_value_error_if_new_item_keys_not_equal_specs(self):
        specs = {
            "1": FixedLenFeature(shape=[1], dtype=torch.int32),
            "2": FixedLenFeature(shape=[1], dtype=torch.int32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = cache_manager.init_cache()
        updates = {
            "1": torch.ones(size=[2, 1], dtype=torch.int32),
        }
        with self.assertRaises(ValueError):
            cache = cache_manager.update_cache(cache, new_items=updates)
        updates = {
            "1": torch.ones(size=[2, 1], dtype=torch.int32),
            "2": torch.ones(size=[2, 1], dtype=torch.int32),
            "3": torch.ones(size=[2, 1], dtype=torch.int32),
        }
        with self.assertRaises(ValueError):
            cache = cache_manager.update_cache(cache, new_items=updates)

    def test_raises_value_error_if_update_item_keys_not_in_specs(self):
        specs = {
            "1": FixedLenFeature(shape=[1], dtype=torch.int32),
            "2": FixedLenFeature(shape=[1], dtype=torch.int32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = cache_manager.init_cache()
        updated_item_data = {
            "1": torch.ones(size=[2, 1], dtype=torch.int32),
            "3": torch.ones(size=[2, 1], dtype=torch.int32),
        }
        updated_item_indices = torch.tensor([0])
        with self.assertRaises(ValueError):
            cache = cache_manager.update_cache(
                cache,
                updated_item_data=updated_item_data,
                updated_item_indices=updated_item_indices,
            )

    def test_masked_update_cache_with_existing_items(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = cache_manager.init_cache()
        updated_item_indices = torch.tensor([1, 3], dtype=torch.int32)
        updated_item_data = {
            "1": torch.tensor([[1, 1], [2, 2]], dtype=torch.int32),
            "2": torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=torch.float32),
        }
        updated_item_mask = torch.tensor([True, False])
        cache = cache_manager.update_cache(
            cache,
            updated_item_data=updated_item_data,
            updated_item_indices=updated_item_indices,
            updated_item_mask=updated_item_mask,
        )
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[0, 0], [1, 1], [0, 0], [0, 0]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [3.0, 3.0, 3.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([1, 0, 1, 1], dtype=torch.int32), cache.age)
        )

    def test_masked_update_cache_with_existing_items_when_all_items_masked(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = negative_cache.NegativeCache(
            data={
                "1": torch.tensor(
                    [[5, 5], [10, 10], [15, 15], [20, 20]], dtype=torch.int32
                )
            },
            age=torch.tensor([2, 2, 2, 2], dtype=torch.int32),
        )
        updated_item_indices = torch.tensor([1, 3], dtype=torch.int32)
        updated_item_data = {
            "1": torch.tensor([[1, 1], [2, 2]], dtype=torch.int32),
        }
        updated_item_mask = torch.tensor([False, False])
        cache = cache_manager.update_cache(
            cache,
            updated_item_data=updated_item_data,
            updated_item_indices=updated_item_indices,
            updated_item_mask=updated_item_mask,
        )
        self.assertEqual({"1"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[5, 5], [10, 10], [15, 15], [20, 20]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([3, 3, 3, 3], dtype=torch.int32), cache.age)
        )

    def test_update_cache_without_lru(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
        }
        cache_manager = negative_cache.CacheManager(
            specs=specs, cache_size=4, use_lru=False
        )
        cache = negative_cache.NegativeCache(
            data={
                "1": torch.tensor(
                    [[5, 5], [10, 10], [15, 15], [20, 20]], dtype=torch.int32
                )
            },
            age=torch.tensor([1, 0, 1, 1], dtype=torch.int32),
        )
        updated_item_indices = torch.tensor([1, 3], dtype=torch.int32)
        updated_item_data = {
            "1": torch.tensor([[1, 1], [2, 2]], dtype=torch.int32),
        }
        cache = cache_manager.update_cache(
            cache,
            updated_item_indices=updated_item_indices,
            updated_item_data=updated_item_data,
        )
        cache_data_expected = torch.tensor(
            [[5, 5], [1, 1], [15, 15], [2, 2]], dtype=torch.int32
        )
        cache_age_expected = torch.tensor([2, 1, 2, 2], dtype=torch.int32)
        self.assertTrue(torch.equal(cache_data_expected, cache.data["1"]))
        self.assertTrue(torch.equal(cache_age_expected, cache.age))

    def test_masked_update_cache_with_existing_items_not_in_index_one(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
            "2": FixedLenFeature(shape=[3], dtype=torch.float32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = cache_manager.init_cache()
        updated_item_indices = torch.tensor([0, 3], dtype=torch.int32)
        updated_item_data = {
            "1": torch.tensor([[1, 1], [2, 2]], dtype=torch.int32),
            "2": torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=torch.float32),
        }
        updated_item_mask = torch.tensor([True, False])
        cache = cache_manager.update_cache(
            cache,
            updated_item_data=updated_item_data,
            updated_item_indices=updated_item_indices,
            updated_item_mask=updated_item_mask,
        )
        self.assertEqual({"1", "2"}, set(cache.data.keys()))
        self.assertTrue(
            torch.equal(
                torch.tensor([[1, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.int32),
                cache.data["1"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    [
                        [3.0, 3.0, 3.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                cache.data["2"],
            )
        )
        self.assertTrue(
            torch.equal(torch.tensor([0, 1, 1, 1], dtype=torch.int32), cache.age)
        )

    def test_new_items_with_mask(self):
        specs = {
            "1": FixedLenFeature(shape=[2], dtype=torch.int32),
        }
        cache_manager = negative_cache.CacheManager(specs=specs, cache_size=4)
        cache = negative_cache.NegativeCache(
            data={
                "1": torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.int32)
            },
            age=torch.tensor([0, 2, 1, 3], dtype=torch.int32),
        )

        new_items = {"1": torch.tensor([[5, 5], [6, 6], [7, 7]], dtype=torch.int32)}
        new_items_mask = torch.tensor([True, False, True])
        cache = cache_manager.update_cache(
            cache, new_items=new_items, new_items_mask=new_items_mask
        )
        self.assertTrue(
            torch.equal(
                torch.tensor([[1, 1], [7, 7], [3, 3], [5, 5]], dtype=torch.int32),
                cache.data["1"],
            )
        )


if __name__ == "__main__":
    unittest.main()
