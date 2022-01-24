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
"""Tests for distributed losses in negative_cache.losses."""

import os
import torch
import unittest
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from negative_cache import losses
from negative_cache import negative_cache


class DistributedCacheClassificationLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.rtol = 1e-4
        self.atol = 1e-4
        mp.set_start_method("fork")

    def init_multiprocessing(self, fn):
        """Initializes multiprocessing for torch.distributed"""
        size = 2
        processes = []
        for rank in range(size):
            p = mp.Process(target=self.init_process, args=(rank, size, fn))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def init_process(self, rank, size, fn, backend="nccl"):
        """Initialize the distributed environment"""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)

    def runInterpretableLoss(self, rank, size):
        """Distributed function"""
        with torch.cuda.device(rank):
            cache_data_multi_replica = {}
            if rank == 0:
                cache_data_multi_replica["data"] = torch.tensor([[1.0], [2.0]]).cuda()
                cache_data_multi_replica["embedding"] = torch.tensor(
                    [[1.0], [2.0]]
                ).cuda()
            else:
                cache_data_multi_replica["data"] = torch.tensor([[-1.0], [-2.0]]).cuda()
                cache_data_multi_replica["embedding"] = torch.tensor(
                    [[-1.0], [-2.0]]
                ).cuda()
            cache_age_multi_replica = torch.zeros(size=[0], dtype=torch.int32).cuda()

            cache = negative_cache.NegativeCache(
                data=cache_data_multi_replica, age=cache_age_multi_replica
            )

            if rank == 0:
                query_embeddings_multi_replica = torch.tensor([[-1.0], [1.0]]).cuda()
                pos_doc_embeddings_multi_replica = torch.tensor([[2.0], [2.0]]).cuda()
            else:
                query_embeddings_multi_replica = torch.tensor([[1.0], [1.0]]).cuda()
                pos_doc_embeddings_multi_replica = torch.tensor([[2.0], [2.0]]).cuda()

            embedding_key = "embedding"
            data_keys = ("data",)

            loss_obj = losses.DistributedCacheClassificationLoss(
                embedding_key=embedding_key, data_keys=data_keys
            )

            def doc_network(data):
                return data["data"]

            def loss_fn_reduced(query_embedding, pos_doc_embedding, cache):
                return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

            output_reduced = loss_fn_reduced(
                query_embeddings_multi_replica,
                pos_doc_embeddings_multi_replica,
                {"cache": cache},
            )

            interpretable_loss_reduced = output_reduced.interpretable_loss
            if rank == 0:
                interpretable_loss_reduced_expected = torch.tensor(
                    [(4.37452 + 0.890350) / 2.0]
                ).cuda()
            else:
                interpretable_loss_reduced_expected = torch.tensor([0.890350]).cuda()

            self.assertTrue(
                torch.isclose(
                    interpretable_loss_reduced_expected,
                    interpretable_loss_reduced,
                    rtol=self.rtol,
                    atol=self.atol,
                ).all()
            )

            loss_obj.reducer = None

            def loss_fn_no_reduce(query_embedding, pos_doc_embedding, cache):
                return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

            output_no_reduce = loss_fn_no_reduce(
                query_embeddings_multi_replica,
                pos_doc_embeddings_multi_replica,
                {"cache": cache},
            )

            interpretable_loss_no_reduce = output_no_reduce.interpretable_loss
            if rank == 0:
                interpretable_loss_no_reduce_expected = torch.tensor(
                    [4.37452, 0.890350]
                ).cuda()
            else:
                interpretable_loss_no_reduce_expected = torch.tensor(
                    [0.890350, 0.890350]
                ).cuda()

            self.assertTrue(
                torch.isclose(
                    interpretable_loss_no_reduce_expected,
                    interpretable_loss_no_reduce,
                    rtol=self.rtol,
                    atol=self.atol,
                ).all()
            )

        dist.destroy_process_group()

    def runTestTrainingLoss(self, rank, size):
        """Distributed function"""
        with torch.cuda.device(rank):
            cache_data_multi_replica = {}
            if rank == 0:
                cache_data_multi_replica["data"] = torch.tensor([[1.0], [2.0]]).cuda()
                cache_data_multi_replica["embedding"] = torch.tensor(
                    [[1.0], [2.0]]
                ).cuda()
            else:
                cache_data_multi_replica["data"] = torch.tensor([[-1.0], [-2.0]]).cuda()
                cache_data_multi_replica["embedding"] = torch.tensor(
                    [[-1.0], [-2.0]]
                ).cuda()
            cache_age_multi_replica = torch.zeros(size=[0], dtype=torch.int32).cuda()

            cache = negative_cache.NegativeCache(
                data=cache_data_multi_replica, age=cache_age_multi_replica
            )

            if rank == 0:
                query_embeddings_multi_replica = torch.tensor([[-1.0], [1.0]]).cuda()
                pos_doc_embeddings_multi_replica = torch.tensor([[2.0], [2.0]]).cuda()
            else:
                query_embeddings_multi_replica = torch.tensor([[1.0], [1.0]]).cuda()
                pos_doc_embeddings_multi_replica = torch.tensor([[2.0], [2.0]]).cuda()

            embedding_key = "embedding"
            data_keys = ("data",)

            loss_obj = losses.DistributedCacheClassificationLoss(
                embedding_key=embedding_key, data_keys=data_keys
            )

            def mock_retrieval_fn(scores):
                if scores.shape[0] == 4:
                    return torch.tensor([[0], [1], [0], [1]], dtype=torch.int64).cuda()
                else:
                    return torch.tensor([[0], [1]], dtype=torch.int64).cuda()

            loss_obj._retrieval_fn = mock_retrieval_fn

            def doc_network(data):
                return data["data"]

            def loss_fn_reduced(query_embedding, pos_doc_embedding, cache):
                return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

            output_reduced = loss_fn_reduced(
                query_embeddings_multi_replica,
                pos_doc_embeddings_multi_replica,
                {"cache": cache},
            )

            training_loss_reduced = output_reduced.training_loss
            if rank == 0:
                training_loss_reduced_expected = torch.tensor(
                    [(0.9874058 + -2.35795) / 2.0]
                ).cuda()
            else:
                training_loss_reduced_expected = torch.tensor(
                    [(-0.589488 + -2.35795) / 2.0]
                ).cuda()

            self.assertTrue(
                torch.isclose(
                    training_loss_reduced_expected,
                    training_loss_reduced,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            )

            loss_obj.reducer = None

            def loss_fn_no_reduce(query_embedding, pos_doc_embedding, cache):
                return loss_obj(doc_network, query_embedding, pos_doc_embedding, cache)

            output_no_reduce = loss_fn_no_reduce(
                query_embeddings_multi_replica,
                pos_doc_embeddings_multi_replica,
                {"cache": cache},
            )

            training_loss_no_reduce = output_no_reduce.training_loss
            if rank == 0:
                training_loss_no_reduce_expected = torch.tensor(
                    [0.9874058, -2.35795]
                ).cuda()
            else:
                training_loss_no_reduce_expected = torch.tensor(
                    [-0.589488, -2.35795]
                ).cuda()

            self.assertTrue(
                torch.isclose(
                    training_loss_no_reduce_expected,
                    training_loss_no_reduce,
                    rtol=self.rtol,
                    atol=self.atol,
                ).all()
            )

        dist.destroy_process_group()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Can't initialize torch.distributed group without CUDA",
    )
    def testInterpretableLoss(self):
        self.init_multiprocessing(self.runInterpretableLoss)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Can't initialize torch.distribute group without CUDA",
    )
    def testTrainingLoss(self):
        self.init_multiprocessing(self.runTestTrainingLoss)


if __name__ == "__main__":
    unittest.main()
