import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"ckpt"))
from ckpt import CheckpointManager,CheckpointError,CFG_SHA,make_objects,prepare_determinism

class CheckpointTests(unittest.TestCase):
    def call(self, directory, steps, *, resume=False, save=False):
        cmd=[sys.executable,str(BASE/"ckpt/ckpt.py"),"train","--directory",str(directory),"--steps",str(steps)]
        if resume: cmd.append("--resume")
        if save: cmd.append("--save")
        cp=subprocess.run(cmd,capture_output=True,text=True,timeout=70)
        if cp.returncode:
            self.fail(f"Checkpoint subprocess failed rc={cp.returncode}\n"
                      f"COMMAND: {cmd}\nSTDERR:\n{cp.stderr[-4000:]}\nSTDOUT:\n{cp.stdout[-2000:]}")
        return json.loads(cp.stdout)

    def test_exact_new_process_continuation_every_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline=self.call(Path(tmp)/"plain",12)
            first=self.call(Path(tmp)/"resume",5,save=True)
            resumed=self.call(Path(tmp)/"resume",12,resume=True)
            self.assertEqual(baseline["records"],first["records"]+resumed["records"])
            self.assertEqual(baseline["state_digest"],resumed["state_digest"])
            self.assertEqual(baseline["rng_digest"],resumed["rng_digest"])
            self.assertEqual(len(resumed["records"]),7)

    def test_corrupt_byte_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.call(tmp,4,save=True)
            cm=CheckpointManager(tmp)
            p=cm.root/cm.pointer_or_none()/"state.pt"
            with p.open("r+b") as f:
                f.seek(15); b=f.read(1); f.seek(15); f.write(bytes([b[0]^0x01]))
            prepare_determinism(); model,opt,sched=make_objects()
            with self.assertRaisesRegex(CheckpointError,"corrupt"):
                cm.load(model,opt,sched,CFG_SHA)

    def test_midwrite_does_not_publish_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.call(tmp,4,save=True)
            cm=CheckpointManager(tmp); before=cm.pointer_or_none()
            prepare_determinism(); model,opt,sched=make_objects()
            with self.assertRaisesRegex(CheckpointError,"INJECTED_FAILURE"):
                cm.save(model,opt,sched,6,6,CFG_SHA,failpoint="before_seal")
            self.assertEqual(cm.pointer_or_none(),before)
            self.assertTrue(list(cm.root.glob("*.tmp")))

    def test_strict_config_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.call(tmp,4,save=True)
            cm=CheckpointManager(tmp)
            prepare_determinism(); model,opt,sched=make_objects()
            with self.assertRaisesRegex(CheckpointError,"Config fingerprint"):
                cm.load(model,opt,sched,"bad_sha")
            self.assertEqual(cm.load(model,opt,sched,CFG_SHA)["step"],4)

if __name__=="__main__":unittest.main()
