from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .hashing import sha256_file
from .io import write_json, read_json


class CheckpointManager:
    def __init__(self,root:str|Path):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.index_path=self.root/"index.json"
        if not self.index_path.exists(): write_json(self.index_path,{"checkpoints":[]})

    def save(self,*,model,optimizer,step:int,next_batch_index:int,ledger_offsets:dict,plan_hash:str,tokenizer_hash:str,parent_checkpoint_hash:str|None=None,branch:str="main") -> dict:
        name=f"{branch}_step_{step:04d}.pt"
        path=self.root/name; tmp=self.root/(name+".tmp")
        payload={
            "schema_version":1,"branch":branch,"global_step":step,"next_batch_index":next_batch_index,
            "ledger_offsets":ledger_offsets,"plan_hash":plan_hash,"tokenizer_hash":tokenizer_hash,
            "parent_checkpoint_hash":parent_checkpoint_hash,
            "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
            "torch_rng_state":torch.get_rng_state(),
        }
        torch.save(payload,tmp); os.replace(tmp,path)
        h=sha256_file(path)
        meta={"path":path.name,"branch":branch,"global_step":step,"next_batch_index":next_batch_index,
              "ledger_offsets":ledger_offsets,"checkpoint_sha256":h,"parent_checkpoint_hash":parent_checkpoint_hash,
              "plan_hash":plan_hash,"tokenizer_hash":tokenizer_hash}
        idx=read_json(self.index_path); idx["checkpoints"].append(meta); write_json(self.index_path,idx)
        write_json(self.root/f"{name}.manifest.json",meta)
        return meta

    def latest(self,branch:str="main")->dict:
        idx=read_json(self.index_path)
        rows=[x for x in idx["checkpoints"] if x["branch"]==branch]
        if not rows: raise RuntimeError(f"no checkpoints for branch {branch}")
        return max(rows,key=lambda x:x["global_step"])

    def at_step(self,step:int,branch:str="main")->dict:
        idx=read_json(self.index_path)
        for x in idx["checkpoints"]:
            if x["branch"]==branch and x["global_step"]==step: return x
        raise KeyError((branch,step))

    def load(self,meta:dict,model,optimizer=None)->dict:
        path=self.root/meta["path"]
        if sha256_file(path)!=meta["checkpoint_sha256"]: raise RuntimeError("checkpoint content hash mismatch")
        payload=torch.load(path,map_location="cpu",weights_only=False)
        if payload["plan_hash"]!=meta["plan_hash"] or payload["tokenizer_hash"]!=meta["tokenizer_hash"]: raise RuntimeError("checkpoint manifest mismatch")
        model.load_state_dict(payload["model_state"])
        if optimizer is not None: optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        return payload
