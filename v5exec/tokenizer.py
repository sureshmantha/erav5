from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .hashing import sha256_json


@dataclass(frozen=True)
class FrozenByteTokenizer:
    """Portable UTF-8 byte tokenizer used only by the execution proof harness.

    Production Bharat V5 keeps the 131,072-token Brahmic-safe tokenizer from the
    mixture specification. This byte tokenizer deliberately avoids tokenizer
    training so the demo has an exactly frozen, platform-independent vocabulary.
    """

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3
    BYTE_OFFSET: int = 4
    VOCAB_SIZE: int = 260
    VERSION: str = "utf8-byte-v1"

    def spec(self) -> dict:
        return {
            "type": self.VERSION,
            "pad_id": self.PAD_ID,
            "bos_id": self.BOS_ID,
            "eos_id": self.EOS_ID,
            "unk_id": self.UNK_ID,
            "byte_offset": self.BYTE_OFFSET,
            "vocab_size": self.VOCAB_SIZE,
            "normalization": "none; exact UTF-8 bytes",
        }

    @property
    def tokenizer_hash(self) -> str:
        return sha256_json(self.spec())

    def encode(self, text: str) -> list[int]:
        return [self.BYTE_OFFSET + b for b in text.encode("utf-8")]

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        data = bytearray()
        for tid in token_ids:
            if tid >= self.BYTE_OFFSET and tid < self.BYTE_OFFSET + 256:
                data.append(tid - self.BYTE_OFFSET)
            elif not skip_special and tid == self.UNK_ID:
                data.extend(b"?")
        return bytes(data).decode("utf-8", errors="strict")

    def token_display(self, token_id: int) -> str:
        if token_id == self.PAD_ID:
            return "<PAD>"
        if token_id == self.BOS_ID:
            return "<BOS>"
        if token_id == self.EOS_ID:
            return "<EOS>"
        if token_id == self.UNK_ID:
            return "<UNK>"
        b = bytes([token_id - self.BYTE_OFFSET])
        try:
            s = b.decode("utf-8")
            return s if s.isprintable() else f"\\x{b[0]:02x}"
        except UnicodeDecodeError:
            return f"\\x{b[0]:02x}"

    def verify_roundtrip(self, texts: Iterable[str]) -> bool:
        return all(self.decode(self.encode(t)) == t for t in texts)
