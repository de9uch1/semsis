from typing import Dict

import pytest
import torch
from torch import Tensor
from transformers import BertModel, BertTokenizerFast

from .sentence_encoder import (
    SentenceEncoder,
    SentenceEncoderAvg,
    SentenceEncoderCls,
    SentenceEncoderSbert,
)

BERT = "bert-base-uncased"
BERT_CLASS = BertModel
BERT_TOKENIZER_CLASS = BertTokenizerFast
D = 768
SBERT = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_D = 384


class SentenceEncoderMock(SentenceEncoder):
    def forward(self, net_inputs: Dict[str, Tensor]) -> Tensor:
        """Return the feature vectors of the given inputs.

        Args:
            net_inputs (Dict[str, Tensor]): The inputs of huggingface models.

        Returns:
            Tensor: Output tensor of shape `(batch_size, embed_dim)`.
        """
        net_outputs = self.model(**net_inputs)
        return net_outputs["last_hidden_state"][:, 0]


@pytest.fixture(scope="module")
def mock_encoder():
    return SentenceEncoderMock(BERT)


@pytest.fixture(scope="module")
def avg_encoder():
    return SentenceEncoderAvg(BERT)


@pytest.fixture(scope="module")
def cls_encoder():
    return SentenceEncoderCls(BERT)


@pytest.fixture(scope="module")
def sbert_encoder():
    return SentenceEncoderSbert(SBERT)


class TestSentenceEncoder:
    def test__init__(self, mock_encoder):
        assert mock_encoder.name_or_path == BERT
        assert (
            isinstance(mock_encoder.model, BERT_CLASS)
            and mock_encoder.model.name_or_path == BERT
        )
        assert (
            isinstance(mock_encoder.tokenizer.tokenizer, BERT_TOKENIZER_CLASS)
            and mock_encoder.tokenizer.tokenizer.name_or_path == BERT
        )
        assert (
            len(list(filter(lambda p: p.requires_grad, mock_encoder.parameters()))) == 0
        )
        assert not mock_encoder.training

    def test_device(self, mock_encoder):
        assert isinstance(mock_encoder.device, torch.device)
        assert mock_encoder.device == next(mock_encoder.parameters()).device

    def test_freeze(self, mock_encoder):
        assert (
            len(list(filter(lambda p: p.requires_grad, mock_encoder.parameters()))) == 0
        )
        assert not mock_encoder.training

    @pytest.mark.parametrize("name_or_path", [BERT, SBERT])
    @pytest.mark.parametrize("representation", ["sbert", "avg", "cls", "dummy"])
    def test_build(self, name_or_path: str, representation: str):
        if representation not in {"avg", "cls", "sbert"}:
            with pytest.raises(NotImplementedError):
                SentenceEncoderMock.build(name_or_path, representation)
            return

        encoder = SentenceEncoderMock.build(name_or_path, representation)
        if representation == "sbert":
            assert isinstance(encoder, SentenceEncoderSbert)
        elif representation == "avg":
            assert isinstance(encoder, SentenceEncoderAvg)
        elif representation == "cls":
            assert isinstance(encoder, SentenceEncoderCls)

    def test_encode(self, mock_encoder):
        examples = ["I like apples.", "This is my pen.", "I like apples."]
        embeddings = mock_encoder.encode(examples)
        assert list(embeddings.shape) == [len(examples), D]
        assert not torch.allclose(embeddings[0], embeddings[1])
        torch.testing.assert_close(embeddings[0], embeddings[2])

    def test_get_embed_dim(self, mock_encoder):
        assert mock_encoder.get_embed_dim() == D


class TestSentenceEncoderAvg:
    def test_forward(self, avg_encoder: SentenceEncoderAvg):
        examples = ["I like apples.", "This is my pen.", "I like apples."]
        batch = avg_encoder.tokenizer(
            examples, return_tensors="pt", padding=True, truncation=True
        ).to(avg_encoder.device)
        embeddings = avg_encoder(batch)
        assert list(embeddings.shape) == [len(examples), D]
        assert not torch.allclose(embeddings[0], embeddings[1])
        torch.testing.assert_close(embeddings[0], embeddings[2])


class TestSentenceEncoderCls:
    def test_forward(self, cls_encoder: SentenceEncoderCls):
        examples = ["I like apples.", "This is my pen.", "I like apples."]
        batch = cls_encoder.tokenizer(
            examples, return_tensors="pt", padding=True, truncation=True
        ).to(cls_encoder.device)
        embeddings = cls_encoder(batch)
        assert list(embeddings.shape) == [len(examples), D]
        assert not torch.allclose(embeddings[0], embeddings[1])
        torch.testing.assert_close(embeddings[0], embeddings[2])


class TestSentenceEncoderSbert:
    def test_forward(self, sbert_encoder: SentenceEncoderSbert):
        examples = ["I like apples.", "This is my pen.", "I like apples."]
        batch = sbert_encoder.tokenizer(
            examples, return_tensors="pt", padding=True, truncation=True
        ).to(sbert_encoder.device)
        embeddings = sbert_encoder(batch)
        assert list(embeddings.shape) == [len(examples), SBERT_D]
        assert not torch.allclose(embeddings[0], embeddings[1])
        torch.testing.assert_close(embeddings[0], embeddings[2])

    def test_get_embed_dim(self, sbert_encoder: SentenceEncoderSbert):
        assert sbert_encoder.get_embed_dim() == SBERT_D
