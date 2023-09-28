import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .tokenizer import Tokenizer

BERT = "bert-base-uncased"
SBERT = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def bert_tokenizer():
    return AutoTokenizer.from_pretrained(BERT)


class TestTokenizer:
    @pytest.mark.parametrize("name_or_path", [BERT, SBERT])
    def test_build(self, name_or_path: str):
        tokenizer = Tokenizer.build(name_or_path)
        assert isinstance(
            tokenizer.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        )

    def test_tokenize(self, bert_tokenizer: PreTrainedTokenizer):
        tokenizer = Tokenizer(bert_tokenizer)
        examples = ["i like apples .", "this is my pen .", "i like apples ."]
        sequences = [tokenizer.tokenize(sentence) for sentence in examples]
        assert len(sequences) == len(examples)
        assert [
            tokenizer.tokenizer.convert_ids_to_tokens(seq) for seq in sequences
        ] == [sentence.split() for sentence in examples]

    def test_collate(self, bert_tokenizer: PreTrainedTokenizer):
        tokenizer = Tokenizer(bert_tokenizer)
        examples = ["i like apples .", "this is my pen .", "i like apples ."]
        sequences = [tokenizer.tokenize(sentence) for sentence in examples]
        batch = tokenizer.collate(sequences)
        expected = tokenizer.tokenizer(
            examples, return_tensors="pt", padding=True, truncation=True
        )
        assert batch.keys() == expected.keys()
        assert [
            torch.equal(v, expected[k]) if torch.is_tensor(v) else v == expected[k]
            for k, v in batch.items()
        ]
