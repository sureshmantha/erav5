#!/usr/bin/env python3
"""GPU admission and diagnostics: one JSON object; PASS/WARN/FAIL/SKIPPED explicit.

Prolog is passive. Full optionally runs an actual NCCL test on an exclusively
allocated node, NEVER alongside production training.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xid_scan import scan as scan_xid

SCHEMA = "v5-health/1"

def check(name, status, reason="", metrics=None, required=False):
    return {"name": name, "status": status, "required": bool(required),
            "reason": reason, "metrics": metrics if metrics is not None else {}}

def command(cmd, timeout=15):
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None

def inspect(mode: str="prolog", expected_gpus: int=0, gpu_required: bool=False,
            xid_source="journalctl", xid_file=None, nccl_timeout=120) -> dict:
    checks=[]; gpu_ids=[]
    binary=shutil.which("nvidia-smi")
    if not binary:
        checks.append(check("gpu_inventory", "skipped", "nvidia-smi not installed", required=True))
    else:
        p=command([binary,"--query-gpu=index,uuid,name,memory.total,memory.free,temperature.gpu",
                   "--format=csv,noheader,nounits"])
        if p is None or p.returncode:
            checks.append(check("gpu_inventory", "fail" if gpu_required else "warn",
                         "nvidia-smi query failed or timed out", required=gpu_required))
        else:
            rows=[r.strip() for r in p.stdout.splitlines() if r.strip()]
            gpu_ids=[r.split(",")[1].strip() for r in rows if "," in r]
            if expected_gpus and len(rows)!=expected_gpus:
                checks.append(check("gpu_inventory","fail",f"expected {expected_gpus}, observed {len(rows)}",
                                    {"count":len(rows), "gpus":rows}, required=True))
            elif gpu_required and not rows:
                checks.append(check("gpu_inventory","fail","No GPUs detected",required=True))
            else:
                checks.append(check("gpu_inventory","pass",metrics={"count":len(rows),"gpus":rows},required=gpu_required))
    # Xid: same fixture invocation can run with zero GPU to prove parser behavior.
    xid=scan_xid(source=xid_source,fixture=xid_file,since_minutes=30)
    checks.append(check("xid",xid["status"],xid["reason"],{"events":xid["events"],"counts":xid["counts"]},required=gpu_required))
    # Optional telemetry. Never turn unsupported ECC into a healthy PASS.
    if binary and gpu_ids:
        ecc=command([binary,"--query-gpu=ecc.errors.uncorrected.volatile.total",
                     "--format=csv,noheader,nounits"])
        if ecc is None or ecc.returncode:
            checks.append(check("ecc_uncorrectable","skipped","ECC query unavailable on this SKU/driver",required=False))
        else:
            vals=[v.strip() for v in ecc.stdout.splitlines() if v.strip()]
            numbers=[]
            for v in vals:
                try: numbers.append(int(v))
                except ValueError: pass
            if len(numbers)!=len(gpu_ids):
                checks.append(check("ecc_uncorrectable","skipped","ECC telemetry not supported on all devices"))
            else:
                checks.append(check("ecc_uncorrectable","fail" if any(x>0 for x in numbers) else "pass",
                                    "uncorrectable ECC count is nonzero" if any(x>0 for x in numbers) else "",
                                    {"values":numbers},required=False))
    else:
        checks.append(check("ecc_uncorrectable","skipped","GPU inventory unavailable"))
    if mode=="full":
        dcgm=shutil.which("dcgmi")
        if not gpu_ids or not dcgm:
            checks.append(check("dcgm_quick", "skipped", "No GPU or dcgmi unavailable"))
        else:
            diag=command([dcgm,"diag","-r","1"],timeout=180)
            if diag is None:
                checks.append(check("dcgm_quick","warn","dcgmi unavailable or timed out"))
            else:
                checks.append(check("dcgm_quick","pass" if diag.returncode==0 else "fail",
                                    "Quick DCGM diagnostic exit code "+str(diag.returncode),
                                    {"exit_code":diag.returncode,"output_tail":diag.stdout[-350:]}))
        if binary and gpu_ids:
            nvlink=command([binary,"nvlink","--status"])
            checks.append(check("nvlink", "pass" if nvlink and nvlink.returncode==0 else "skipped",
                                "NVLink status unavailable or not supported" if not nvlink or nvlink.returncode else ""))
        if not gpu_ids:
            checks.append(check("nccl_all_reduce","skipped","No GPU for NCCL test",required=gpu_required))
        elif len(gpu_ids) < 2:
            checks.append(check("nccl_all_reduce","skipped","Need at least two allocated GPUs for inter-GPU collective proof",required=True))
        elif not shutil.which("torchrun"):
            checks.append(check("nccl_all_reduce","skipped","torchrun not installed",required=True))
        else:
            env=os.environ.copy(); env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING","1")
            cmd=["torchrun","--standalone",f"--nproc-per-node={len(gpu_ids)}",
                 str(Path(__file__).parent/"nccl_probe.py"),"--backend","nccl","--sizes","1024,1048576,8388608"]
            try: p=subprocess.run(cmd,capture_output=True,text=True,timeout=nccl_timeout,env=env)
            except (OSError,subprocess.TimeoutExpired) as exc:
                checks.append(check("nccl_all_reduce","fail",f"NCCL execution timeout/error: {exc}",required=True))
            else:
                parsed=None
                for line in reversed(p.stdout.splitlines()):
                    try: parsed=json.loads(line); break
                    except json.JSONDecodeError: continue
                if p.returncode!=0 or not isinstance(parsed,dict) or parsed.get("status")!="pass":
                    checks.append(check("nccl_all_reduce","fail","NCCL incorrect / failed; inspect stderr",
                                        {"stderr_tail":p.stderr[-500:],"exit_code":p.returncode},required=True))
                else:
                    checks.append(check("nccl_all_reduce","pass",metrics=parsed,required=True))
    failing=[c for c in checks if c["status"]=="fail"]
    uncertain=[c for c in checks if c["status"]=="warn" or (c["status"]=="skipped" and c["required"])]
    verdict="fail" if failing else "warn" if uncertain else "pass"
    return {"schema":SCHEMA,"host":socket.gethostname(),"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "mode":mode,"expected_gpus":expected_gpus,"verdict":verdict,
            "admit_training": verdict=="pass" and (not gpu_required or len(gpu_ids)>0),
            "checks":checks,"drain_reason":"; ".join(x["name"]+": "+x["reason"] for x in failing) or None}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("prolog","full"),default="prolog")
    p.add_argument("--expected-gpus",type=int,default=0)
    p.add_argument("--require-gpu",action="store_true")
    p.add_argument("--xid-source",choices=("journalctl","dmesg","file"),default="journalctl")
    p.add_argument("--xid-file")
    p.add_argument("--nccl-timeout",type=int,default=120)
    p.add_argument("--out",help="Atomically write a JSON copy, stdout still emits JSON")
    a=p.parse_args()
    result=inspect(a.mode,a.expected_gpus,a.require_gpu,a.xid_source,a.xid_file,a.nccl_timeout)
    line=json.dumps(result,sort_keys=True)
    if a.out:
        dest=Path(a.out); dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_name(dest.name+".tmp."+str(os.getpid()))
        tmp.write_text(line+"\n"); os.replace(tmp,dest)
    print(line)
    return 2 if result["verdict"]=="fail" else 1 if result["verdict"]=="warn" else 0
if __name__=="__main__": sys.exit(main())
