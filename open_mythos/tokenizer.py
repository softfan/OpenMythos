from __future__ import annotations

from typing import Iterable, List, Sequence

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    Args:
        model_id: HuggingFace model ID or local tokenizer path.
                  Defaults to "openai/gpt-oss-20b".

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> text = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MythosTokenizer requires transformers. Install with: "
                "pip install transformers"
            ) from exc

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Generation/training utilities often expect pad_token to exist.
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return int(self.tokenizer.vocab_size)

    @property
    def eos_token_id(self) -> int | None:
        """Return EOS token id, if available."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Return PAD token id, if available."""
        return self.tokenizer.pad_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text into token ids."""
        return list(
            self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        )

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids into text."""
        return str(
            self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
            )
        )

    def batch_encode(
        self,
        texts: Iterable[str],
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ):
        """Batch-encode an iterable of strings."""
        return self.tokenizer(
            list(texts),
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    def batch_decode(
        self,
        batch_token_ids,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Batch-decode token id sequences."""
        return list(
            self.tokenizer.batch_decode(
                batch_token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        )


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID) -> MythosTokenizer:
    """Convenience factory used by open_mythos.__init__.__all__."""
    return MythosTokenizer(model_id=model_id)


def get_vocab_size(model_id: str = DEFAULT_MODEL_ID) -> int:
    """Load tokenizer and return its vocabulary size."""
    return load_tokenizer(model_id).vocab_size
