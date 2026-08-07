from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def _write(self, level: str, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        line = f"{ts} [{level}] {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def pass_(self, event: str, detail: str = "") -> None:
        self._write("PASS", event + (f" {detail}" if detail else ""))

    def warn(self, message: str) -> None:
        self._write("WARN", message)

    def fail(self, event: str, detail: str = "") -> None:
        self._write("FAIL", event + (f" {detail}" if detail else ""))
