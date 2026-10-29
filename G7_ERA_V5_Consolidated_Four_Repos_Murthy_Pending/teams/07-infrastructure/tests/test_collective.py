"""Local 2-rank CPU Gloo acceptance without a torchrun rendezvous subprocess.

Launch exactly two child interpreters with a loopback TCP rendezvous. This tests
Gloo collective correctness, NOT GPU NCCL or multi-node behavior.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

BASE = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class CollectiveTests(unittest.TestCase):
    def test_2rank_cpu_gloo_exact_collective(self):
        port = free_port()
        base_env = dict(os.environ)
        base_env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                        WORLD_SIZE="2", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        PYTHONUNBUFFERED="1")
        # Mac Docker / VPN host-name resolution can break --standalone/c10d.
        # Loopback-interface selection reduces nondeterminism on local macOS.
        if sys.platform == "darwin":
            base_env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        children = []
        try:
            for rank in range(2):
                env = dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank))
                children.append(subprocess.Popen(
                    [sys.executable, str(BASE / "health/nccl_probe.py"),
                     "--backend", "gloo", "--sizes", "1024,65536"],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True))
            outputs = []
            deadline = time.monotonic() + 45
            for rank, child in enumerate(children):
                timeout = max(1, deadline - time.monotonic())
                try:
                    out, err = child.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    for c in children:
                        if c.poll() is None:
                            c.kill()
                    for c in children:
                        c.communicate()
                    self.fail(f"Rank {rank} timed out; localhost rendezvous port={port}. "
                              "Check firewall/VPN and GLOO_SOCKET_IFNAME.")
                outputs.append((out, err, child.returncode))
            self.assertTrue(all(rc == 0 for _, _, rc in outputs),
                            msg=f"Child diagnostics: {outputs}")
            records = []
            for out, _, _ in outputs:
                records.extend(json.loads(line) for line in out.splitlines()
                               if line.strip().startswith("{"))
            self.assertEqual(len(records), 1, msg=str(outputs))
            self.assertEqual(records[0]["backend"], "gloo")
            self.assertEqual(records[0]["world_size"], 2)
            self.assertTrue(all(x["all_ranks_correct"] for x in records[0]["sizes"]))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate()

if __name__ == "__main__":
    unittest.main()
