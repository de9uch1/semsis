import abc
from typing import Dict, List, Literal

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel

from .tokenizer import Tokenizer


class SentenceEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Huggingface model wrapper.

    Args:
        name_or_path (str): Model name or path.
    """

    def __init__(self, name_or_path: str) -> None:
        super().__init__()
        self.name_or_path = name_or_path
        self.load_model(name_or_path)
        self.freeze()

    @property
    def device(self) -> torch.device:
        """Return the device of this model."""
        return self.model.device

    def load_model(self, name_or_path: str) -> None:
        """Load the model and tokenizer.

        Args:
            name_or_path (str): Model name or path.
        """
        self.model = AutoModel.from_pretrained(name_or_path)
        self.tokenizer = Tokenizer.build(name_or_path)

    def freeze(self) -> None:
        for p in self.parameters():
            if getattr(p, "requires_grad", None) is not None:
                p.requires_grad = False
        self.eval()

    @classmethod
    def build(
        cls,
        model_name_or_path: str,
        representation: Literal["avg", "cls", "sbert"],
    ) -> "SentenceEncoder":
        """Build the sentence encoder model.

        Args:
            model_name_or_path (str): Model name or path of huggingface transformer models.
            representation (Literal["avg", "cls", "sbert"]): Type of the sentence representation.

        Returns:
            SentenceEncoder: This class.
        """
        if representation == "sbert":
            return SentenceEncoderSbert(model_name_or_path)
        elif representation == "avg":
            return SentenceEncoderAvg(model_name_or_path)
        elif representation == "cls":
            return SentenceEncoderCls(model_name_or_path)
        else:
            raise NotImplementedError(f"`{representation}` is not supported.")

    def encode(self, sentences: List[str]) -> Tensor:
        """Encode sentences into their sentence vectors.

        Args:
            sentences (List[str]): Input sentences.

        Returns:
            Tensor: Sentence vectors of shape `(batch_size, embed_dim)`.
        """
        batch = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        return self.forward(batch)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Return the feature vectors of the given inputs.

        Returns:
            Tensor: Output tensor of shape `(batch_size, embed_dim)`.
        """

    def get_embed_dim(self) -> int:
        """Get the size of the embedding dimension.

        Returns:
            int: The size of the embedding dimension.
        """
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        return self.model.get_input_embeddings().embedding_dim


class SentenceEncoderAvg(SentenceEncoder):
    def forward(self, net_inputs: Dict[str, Tensor]) -> Tensor:
        """Return the feature vectors of the given inputs.

        Args:
            net_inputs (Dict[str, Tensor]): The inputs of huggingface models.

        Returns:
            Tensor: Output tensor of shape `(batch_size, embed_dim)`.
        """
        net_outputs = self.model(**net_inputs)
        non_pad_mask = net_inputs["attention_mask"]
        active_hiddens = net_outputs["last_hidden_state"] * non_pad_mask.unsqueeze(-1)
        return active_hiddens.sum(dim=1) / non_pad_mask.sum(dim=1, keepdim=True)


class SentenceEncoderCls(SentenceEncoder):
    def forward(self, net_inputs: Dict[str, Tensor]) -> Tensor:
        """Return the feature vectors of the given inputs.

        Args:
            net_inputs (Dict[str, Tensor]): The inputs of huggingface models.

        Returns:
            Tensor: Output tensor of shape `(batch_size, embed_dim)`.
        """
        net_outputs = self.model(**net_inputs)
        return net_outputs["pooler_output"]


class SentenceEncoderSbert(SentenceEncoder):
    def load_model(self, name_or_path: str) -> None:
        """Load the model and tokenizer.

        Args:
            name_or_path (str): Model name or path.
        """
        self.model = SentenceTransformer(name_or_path)
        self.tokenizer = Tokenizer(self.model.tokenizer)

    def forward(self, net_inputs: Dict[str, Tensor]) -> Tensor:
        """Return the feature vectors of the given inputs.

        Args:
            net_inputs (Dict[str, Tensor]): The inputs of huggingface models.

        Returns:
            Tensor: Output tensor of shape `(batch_size, embed_dim)`.
        """
        net_outputs = self.model(net_inputs)
        return net_outputs["sentence_embedding"]

    def get_embed_dim(self) -> int:
        """Get the size of the embedding dimension.

        Returns:
            int: The size of the embedding dimension.
        """
        return self.model.get_sentence_embedding_dimension()
