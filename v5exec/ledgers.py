from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .io import append_jsonl, append_many_jsonl, count_jsonl, read_jsonl


class Ledgers:
    def __init__(self, root: str | Path):
        root=Path(root); root.mkdir(parents=True,exist_ok=True)
        self.consumption=root/"consumption.jsonl"
        self.learning=root/"learning.jsonl"
        self.token_learning=root/"token_learning.jsonl"
        self.opus=root/"opus_decisions.jsonl"

    def offsets(self)->dict:
        return {"consumption":count_jsonl(self.consumption),"learning":count_jsonl(self.learning),"token_learning":count_jsonl(self.token_learning),"opus":count_jsonl(self.opus)}

    def append_opus(self,row:dict)->int:
        return append_jsonl(self.opus,row)

    def append_consumption(self,row:dict)->int:
        return append_jsonl(self.consumption,row)

    def append_learning(self,row:dict)->int:
        return append_jsonl(self.learning,row)

    def append_token_learning_rows(self,rows:list[dict])->int:
        return append_many_jsonl(self.token_learning, rows)

    def consumption_rows(self): return read_jsonl(self.consumption)
    def learning_rows(self): return read_jsonl(self.learning)
