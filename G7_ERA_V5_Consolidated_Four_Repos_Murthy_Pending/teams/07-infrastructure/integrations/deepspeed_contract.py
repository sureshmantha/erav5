#!/usr/bin/env python3
"""Framework-adapter contract; validation is executable without DeepSpeed.

Does not pretend DeepSpeed is installed or that its ZeRO checkpoint is a remote
transaction. Actual save/load paths must be integration-tested with Team 5 trainer
and DeepSpeed engine across real ranks before production adoption.
"""
from __future__ import annotations
import argparse,json,pathlib,sys
class ConfigurationError(ValueError):pass

def validate(config):
    if not isinstance(config,dict):raise ConfigurationError('expected JSON object')
    if not isinstance(config.get('zero_optimization'),dict):raise ConfigurationError('zero_optimization object required')
    stage=config['zero_optimization'].get('stage')
    if not isinstance(stage,int) or stage not in (0,1,2,3):raise ConfigurationError('ZeRO stage must be 0..3')
    micro=config.get('train_micro_batch_size_per_gpu');acc=config.get('gradient_accumulation_steps')
    if any(not isinstance(x,int) or x<1 for x in (micro,acc)):raise ConfigurationError('microbatch and accumulation must be positive integers')
    world=config.get('g7_expected_data_parallel_ranks')
    if not isinstance(world,int) or world<1:raise ConfigurationError('g7_expected_data_parallel_ranks required for validation')
    configured=config.get('train_batch_size')
    expected=micro*acc*world
    if configured is not None and configured!=expected:raise ConfigurationError(f'global batch mismatch configured={configured} expected={expected}; DP ranks, not all TP/PP ranks')
    if config.get('fp16',{}).get('enabled',False) and config.get('bf16',{}).get('enabled',False):raise ConfigurationError('fp16 and bf16 cannot both be enabled')
    return {'status':'PASS_STATIC_CONFIG','zero_stage':stage,'expected_data_parallel_ranks':world,'global_microbatch_examples':expected,
            'limitations':['DeepSpeed not initialized','GPU/NCCL not tested','ZeRO checkpoint save/load not tested']}

def save_engine_checkpoint_all_ranks(engine,root,tag,client_state):
    """Training integration hook: MUST be invoked on all model ranks.

    NOT a complete remote transaction. Caller must verify output, all-rank receipts,
    remote durable availability, and fenced manifest before publishing latest.
    """
    if not isinstance(client_state,dict) or not {'run_id','global_step','data_cursor','config_sha'}<=set(client_state):
        raise ConfigurationError('client_state missing run/step/data/config identity')
    if not isinstance(tag,str) or not tag or '/' in tag or '\\' in tag:raise ConfigurationError('unsafe tag')
    return engine.save_checkpoint(str(root),tag=tag,client_state=client_state,save_latest=False)

def restore_engine_checkpoint_all_ranks(engine,root,tag,expected_config_sha):
    """Trainer must be reinitialized in the right ZeRO stage before invoking."""
    load_path, extra=engine.load_checkpoint(str(root),tag=tag,load_module_strict=True,
                                              load_optimizer_states=True,load_lr_scheduler_states=True)
    if not load_path:raise RuntimeError('DeepSpeed load returned no checkpoint')
    if extra is None or extra.get('config_sha')!=expected_config_sha:raise RuntimeError('config/provenance mismatch')
    return extra

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    obj=json.loads(pathlib.Path(a.config).read_text());print(json.dumps(validate(obj),indent=2))
if __name__=='__main__':
    try:main()
    except (ConfigurationError,OSError,ValueError) as e:print(f'CONFIG_ERROR: {e}',file=sys.stderr);sys.exit(2)
