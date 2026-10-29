#!/usr/bin/env python3
"""Real torch.distributed collective proof. Requires torchrun with N ranks.

Do not present CPU/gloo results as NCCL GPU results.
"""
import argparse
from datetime import timedelta
import json
import os
import sys
import time
import torch
import torch.distributed as dist

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=("nccl","gloo"),required=True)
    p.add_argument("--sizes",default="1024,1048576")
    args=p.parse_args()
    local=int(os.environ["LOCAL_RANK"]); rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
    if args.backend=="nccl":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA absent: cannot run NCCL")
        torch.cuda.set_device(local); device=torch.device("cuda",local)
    else: device=torch.device("cpu")
    if args.backend == "gloo":
        # A small CPU fixture should not monopolize all host cores per rank.
        torch.set_num_threads(1)
    dist.init_process_group(args.backend,timeout=timedelta(seconds=25))
    outcomes=[]
    try:
        for bytes_count in [int(v) for v in args.sizes.split(",")]:
            n=max(1,bytes_count//4); t=torch.full((n,),float(rank+1),device=device)
            dist.barrier(); start=time.perf_counter()
            dist.all_reduce(t,op=dist.ReduceOp.SUM)
            if device.type=="cuda": torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            expected=float(world*(world+1)//2)
            deviation=float((t-expected).abs().max().item())
            ok=bool(deviation==0)
            validity=torch.tensor([1 if ok else 0],device=device,dtype=torch.int32)
            dist.all_reduce(validity,op=dist.ReduceOp.MIN)
            outcomes.append({"bytes":n*4,"seconds":elapsed,"max_error":deviation,"all_ranks_correct":bool(validity.item())})
        if rank==0:
            print(json.dumps({"schema":"g7-collective/1","backend":args.backend,"world_size":world,
                              "status":"pass" if all(x["all_ranks_correct"] for x in outcomes) else "fail",
                              "sizes":outcomes},sort_keys=True),flush=True)
        if not all(x["all_ranks_correct"] for x in outcomes): return 2
        return 0
    finally: dist.destroy_process_group()
if __name__=="__main__":
    try: sys.exit(main())
    except Exception as exc:
        print(f"COLLECTIVE_ERROR {exc}",file=sys.stderr); sys.exit(2)
