from __future__ import annotations

from typing import Union

from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class Tokenizer:
    def __init__(
        self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    ) -> None:
        self.tokenizer = tokenizer

    @property
    def is_fast(self) -> bool:
        """Whether the tokenizer is `PreTrainedTokenizerFast` or not."""
        return self.tokenizer.is_fast

    @classmethod
    def build(cls, model_name_or_path: str) -> Tokenizer:
        """Build the tokenizer.

        Args:
            model_name_or_path (str): Model name or path of huggingface transformer models.

        Returns:
            Tokenizer: This class.
        """
        return cls(AutoTokenizer.from_pretrained(model_name_or_path))

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def tokenize(self, sentence: str) -> list[int]:
        """Tokenize and convert a sentence to the token sequence.

        Args:
            sentence (str): An input sentence.

        Returns:
            list[int]: A token ID sequence.
        """
        return self.tokenizer.encode(
            sentence, add_special_tokens=False, truncation=True
        )

    def tokenize_batch(self, sentences: list[str]) -> list[list[int]]:
        """Tokenize and convert sentences to their token sequences.

        Args:
            sentences (list[str]): Input sentences.

        Returns:
            list[list[int]]: Token ID sequences.
        """
        if self.is_fast:
            return self.tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=False,
                truncation=True,
                return_attention_mask=False,
            )["input_ids"]
        else:
            return [self.tokenize(sentence) for sentence in sentences]

    def collate(self, samples: list[list[int]]) -> BatchEncoding:
        """Make a mini-batch from samples.

        Args:
            samples (list[list[int]]): Token sequences.

        Returns:
            BatchEncoding: A mini-batch.
        """
        batch = {}
        for sample in samples:
            item = self.tokenizer.prepare_for_model(
                sample,
                None,
                add_spenical_tokens=True,
                padding=False,
                truncation=True,
                pad_to_multiple_of=None,
                return_attention_mask=False,
                return_tensors=None,
            )
            for key, value in item.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        return self.tokenizer.pad(batch, padding=True, return_tensors="pt")
