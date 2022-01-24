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
"""Functions for calculating masks of new cache items."""
import abc
import torch


class CacheFilterFn(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, cache, new_items):
        pass


class IsInCacheFilterFn(CacheFilterFn):
    """Creates a mask for items that are already in the cache.

    Given a tuple of keys, this class is a function that checks if there is a
    cache element that matches exactly on all keys.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, cache, new_items):
        datawise_matches = []
        for key in self.keys:
            cache_vals = cache.data[key]
            new_items_vals = new_items[key]
            if cache_vals.dtype.is_floating_point:
                raise NotImplementedError("Floating datatypes are not yet implemented.")
            cache_vals = torch.unsqueeze(cache_vals, dim=0)
            new_items_vals = torch.unsqueeze(new_items_vals, dim=1)
            elementwise = cache_vals == new_items_vals
            i = 2
            while i < elementwise.dim():
                datawise = torch.all(elementwise, dim=-1)
                i += 1
            datawise_matches.append(datawise)
        all_keys_datawise = torch.stack(datawise_matches, dim=2)
        all_keys_match = torch.all(all_keys_datawise, dim=2)
        in_cache = torch.any(all_keys_match, dim=1)
        return ~in_cache
