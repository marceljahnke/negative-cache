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
"""A collection of retrieval functions for negative mining.

Retrieval functions take in a matrix of scores and return a batch x `k` set of
indices indicating the `k` items retrieved.
"""
import abc
import torch


class AbstractRetrievalFn(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, scores):
        pass


class MaxScoreRetrievalFn(AbstractRetrievalFn):
    def __call__(self, scores):
        indices = torch.argmax(scores, dim=1)
        return torch.unsqueeze(indices, dim=1)


def _sample_gumbel(shape):
    uniform_vals = torch.rand(shape)
    gumbel_vals = -torch.log(-torch.log(uniform_vals))
    return gumbel_vals


class GumbelMaxRetrievalFn(AbstractRetrievalFn):
    """Creates a retrieval function that uses Gumbel-max sampling.

    Gumbel-max sampling is an approach to sample from the softmax distribution of
    a set of scores by perturbing the scores then taking the argmax. The scores
    are first scaled by `inv_temp` then perturbed by adding Gumbel noise.
    """

    def __init__(self, inv_temp=1.0):
        super(GumbelMaxRetrievalFn, self).__init__()
        self.inv_temp = inv_temp

    def __call__(self, scores):
        device = scores.device
        # scores: [batch_size x top_k] or [batch_size x cache_size]
        gumbel_vals = _sample_gumbel(scores.size())
        # gumbel values of same shape as scores
        perturbed_scores = self.inv_temp * scores + gumbel_vals.to(device)
        indices = torch.argmax(perturbed_scores, dim=1)
        # indices: [batch_size] ==> indices of highest perturbed_score per batch
        # change shape for return to: [batch_size] --> [batch_size x 1]
        return torch.unsqueeze(indices, dim=1)
