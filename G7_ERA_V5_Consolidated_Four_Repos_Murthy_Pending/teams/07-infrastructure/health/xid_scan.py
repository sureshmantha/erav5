#!/usr/bin/env python3
"""Read Linux NVIDIA Xid events. JSON-only stdout; errors distinguish SKIPPED from PASS.

The classification is a conservative triage suggestion, not a universal vendor
hardware diagnosis. Review the NVIDIA Xid catalog for each GPU/driver generation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import re
import subprocess
import sys

# Conservative admission decisions. 94 is contained and often restart_app, not always drain.
HARD = {48: "uncorrectable ECC (DBE)", 64: "row-remapping failure",
        74: "NVLink hardware/connection error", 79: "fallen off bus",
        95: "uncontained memory error", 119: "GSP RPC timeout", 120: "GSP error"}
WARN = {13: "graphics engine exception", 31: "MMU page fault (often application)",
        43: "application stopped (GPU may remain healthy)",
        45: "preemptive channel cleanup", 63: "row-remap event: investigate",
        94: "contained memory error: restart affected app"}
# Example: NVRM: Xid (PCI:0000:4b:00): 79, ... or NVRM: Xid (0000:4b:00): 79
PATTERN = re.compile(r"NVRM:\s*Xid\s*\((?:PCI:)?\s*([0-9a-fA-F:. ]+)\)\s*:\s*(\d+)\b", re.I)

def collect(source: str, fixture: str | None, since_minutes: int):
    if source == "file":
        if not fixture:
            return None, "--file is required with --source=file"
        try:
            with open(fixture, encoding="utf-8", errors="replace") as fp: return fp.read(), None
        except OSError as exc: return None, f"fixture unreadable: {exc}"
    if source == "journalctl":
        cmd = ["journalctl", "-k", "--no-pager", "--since", f"-{since_minutes} minutes", "-o", "short-iso"]
    else:
        cmd = ["dmesg"] # No time window: pre-existing errors may be reported.
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=20, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return None, f"cannot run {cmd[0]}: {exc}"
    if proc.returncode:
        return None, f"{cmd[0]} returned {proc.returncode}: {proc.stderr.strip()[:160]}"
    return proc.stdout, None

def analyze(log: str):
    entries = []
    for line in log.splitlines():
        m = PATTERN.search(line)
        if not m: continue
        bdf = m.group(1).replace(" ", ""); code = int(m.group(2))
        level = "fail" if code in HARD else "warn"
        entries.append({"pci_bdf": bdf, "code": code, "severity": level,
                        "reason": HARD.get(code, WARN.get(code, "unclassified Xid: investigate"))})
    counts = dict(sorted(Counter(str(e["code"]) for e in entries).items(), key=lambda x:int(x[0])))
    status = "fail" if any(e["severity"] == "fail" for e in entries) else "warn" if entries else "pass"
    return status, entries, counts

def scan(source="journalctl", fixture=None, since_minutes=30):
    log, err = collect(source, fixture, since_minutes)
    result = {"name": "xid", "status": "skipped", "source": source,
              "since_minutes": since_minutes if source == "journalctl" else None,
              "reason": err or "", "events": [], "counts": {}}
    if err: return result
    status, events, counts = analyze(log)
    result.update(status=status, events=events, counts=counts, reason=("full kernel buffer, no time window" if source=="dmesg" else ""))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source", choices=("journalctl","dmesg","file"),default="journalctl")
    p.add_argument("--file")
    p.add_argument("--since-minutes",type=int,default=30)
    args=p.parse_args()
    if args.since_minutes<=0: p.error("--since-minutes must be positive")
    result=scan(args.source,args.file,args.since_minutes)
    result["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
    print(json.dumps(result,sort_keys=True))
    return 2 if result["status"]=="fail" else 1 if result["status"] in ("warn","skipped") else 0
if __name__=="__main__": sys.exit(main())
