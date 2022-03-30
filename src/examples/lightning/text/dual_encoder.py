from typing import Tuple
import torch
from pytorch_lightning import LightningModule
from transformers import AutoModel

from negative_cache.handlers import CacheLossHandler
from negative_cache.losses import (
    CacheClassificationLoss,
    DistributedCacheClassificationLoss,
)
from negative_cache.negative_cache import CacheManager, FixedLenFeature

"""
This class provides an overview of how to use the negative_cache library. It is not intended for standalone execution.
"""


class DualEncoder(LightningModule):
    def __init__(self):
        self.query_encoder = AutoModel.from_pretrained(
            "bert-base-uncased", return_dict=True
        )
        self.document_encoder = AutoModel.from_pretrained(
            "bert-base-uncased", return_dict=True
        )

        self.handler = self.setup_negative_cache(cache_size=131072, top_k=64)

    def setup_negative_cache(
        self,
        cache_size: int,
        top_k: int,
        feature_size: int = 512,
        emb_size: int = 768,
        distributed_cache_loss: bool = False,
    ) -> CacheLossHandler:
        """
        Constructs the negative cache with the specified parameters.

        Args:
            cache_size (int): The cache size, specifies the number of items (i.e. passages or documents) that will be cached.
            top_k (int): Number of approximated top scores computed for each query in the batch, which are used for gumble max sampling.
            feature_size (int): Number of features per item in the negative cache.
            emb_size (int): Number of dimensions of the item embedding to be stored in the negative cache.
            distributed_sync (bool): Should the negative caches be synchronized during distributed training (at the end of each iteration). Requires NCCL as the backend.

        Returns:
            CacheLossHandler: All communication with the negative cache is done via this handler object. Only call update_cache_and_compute_loss at training step.
        """
        data_keys = (
            "input_ids",
            "attention_mask",
        )
        embedding_key = "embedding"
        specs = {
            "input_ids": FixedLenFeature(shape=[feature_size], dtype=torch.int32),
            "attention_mask": FixedLenFeature(shape=[feature_size], dtype=torch.int32),
            "embedding": FixedLenFeature(shape=[emb_size], dtype=torch.float32),
        }
        cache_manager = CacheManager(specs, cache_size=cache_size)
        if distributed_cache_loss:
            cache_loss = DistributedCacheClassificationLoss(
                embedding_key=embedding_key,
                data_keys=data_keys,
                score_transform=lambda score: 20 * score,
                top_k=top_k,
            )
        else:
            cache_loss = CacheClassificationLoss(
                embedding_key=embedding_key,
                data_keys=data_keys,
                score_transform=lambda score: 20
                * score,  # Optional, applied to scores before loss.
                top_k=top_k,  # NOTE: cache_size % top_k == 0 for top-k approximation!
            )
        handler = CacheLossHandler(
            cache_manager=cache_manager,
            cache_loss=cache_loss,
            embedding_key=embedding_key,
            data_keys=data_keys,
        )
        return handler

    def encode_query(self, data) -> torch.Tensor:
        """Computes the query embeddings.

        Args:
            data (Dict): Dictionary consisting of the input_ids, token_type_ids and attention_mask as provided by the used tokenizer.

        Returns:
            torch.Tensor: [Batch Size x Embedding Size] shaped tensor containg the document embeddings produced by the query network.
        """
        if data is None:
            return None

        # for this example, we use the [CLS] token representation
        out = self.query_encoder(**data, return_dict=True)
        hidden = out.last_hidden_state
        return hidden[:, 0]

    def encode_doc(self, data) -> torch.Tensor:
        """Computes the document embeddings.

        Args:
            data (Dict): Dictionary consisting of the input_ids, token_type_ids and attention_mask as provided by the used tokenizer.

        Returns:
            torch.Tensor: [Batch Size x Embedding Size] shaped tensor containg the document embeddings produced by the document network.
        """
        if data is None:
            return None

        # for this example, we use the [CLS] token representation
        out = self.document_network(**data, return_dict=True)
        hidden = out.last_hidden_state
        return hidden[:, 0]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Computes one training step. First the embeddings of the queries and documents from the current batch are computed, 
        then the negative cache is used to calculate the (distributed) cache classification loss.

        Args:
            batch (_type_): For this example the batch data is a Tuple containing two Dictionaries. Each Dictionary consists of the input_ids, token_type_ids and attention_mask as provided by the used tokenizer.
            batch_idx (int): The batch index.

        Returns:
            torch.Tensor: The gradient of the cache cross entropy loss for one training step. Used to update the model parameters.
        """
        query_data, doc_data = batch
        q_emb = self.encode_query(query_data)
        d_emb = self.encode_doc(doc_data)

        # move the embedding tensor of the (sharded) negative cache to the correct device
        self.handler.cache.data["embedding"] = self.handler.cache.data["embedding"].to(
            self.device
        )

        # use the negative cache
        loss = self.handler.update_cache_and_compute_loss(
            item_network=self.encode_doc,
            query_embeddings=q_emb,
            pos_item_embeddings=d_emb,
            features=doc_data,
            writer=(self.logger.experiment, self.global_step),
        )

        self.log("train_loss", loss)
        return loss

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # one possible implementation of the forward method
        query_data, doc_data = batch
        q_emb = self.encode_query(query_data)
        d_emb = self.encode_doc(doc_data)
        scores = None
        if q_emb is not None and d_emb is not None:
            scores = torch.matmul(q_emb, d_emb.T).to(self.device)
        return (q_emb, d_emb, scores)
