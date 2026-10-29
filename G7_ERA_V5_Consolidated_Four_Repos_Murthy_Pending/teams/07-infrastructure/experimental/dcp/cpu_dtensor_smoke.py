#!/usr/bin/env python3
"""Actual CPU 2-rank DTensor DCP save and restore; NOT an optimizer/train-run test.

A distributed state **format compatibility** experiment. Requires torch with DTensor+DCP,
loopback networking, and a shared POSIX directory. Does not prove GPU, S3, resharding
of DeepSpeed ZeRO optimizer states, or crash-atomic multi-rank commit.
"""
import argparse, json, os, pathlib, sys
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['save','load'],required=True);p.add_argument('--root',required=True);p.add_argument('--world-size',type=int,default=2)
    args=p.parse_args();dist.init_process_group('gloo');rank=dist.get_rank()
    assert args.world_size==dist.get_world_size()
    mesh=init_device_mesh('cpu',(args.world_size,))
    full=torch.arange(48,dtype=torch.float64).reshape(12,4)
    if args.mode=='load':full*=0
    dt=distribute_tensor(full,mesh,[Shard(0)])
    state={'model.weight':dt}
    path=str(pathlib.Path(args.root).resolve())
    try:
        if args.mode=='save': dcp.save(state,checkpoint_id=path)
        else:
            dcp.load(state,checkpoint_id=path)
            local=state['model.weight'].to_local()
            restored=state['model.weight'].full_tensor()
            expected=torch.arange(48,dtype=torch.float64).reshape(12,4)
            if not torch.equal(restored,expected):raise AssertionError('DTensor restore did not match')
            if rank==0: print(json.dumps({'result':'PASS','scope':'cpu DTensor DCP saved and loaded','world_size':args.world_size,'tensor_digest_equal':True}))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
