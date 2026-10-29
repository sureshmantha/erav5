#!/usr/bin/env python3
"""Actual CPU training/restore + *simulated* scheduler and hardware event policy.

This uses the real G7 ckpt.py exact CPU training fixture as separate OS processes;
Slurm, GPU devices, node faults and S3 commits are NOT executed.
"""
import argparse,json,pathlib,subprocess,sys,tempfile,hashlib,datetime
BASE=pathlib.Path(__file__).resolve().parents[1]
ENGINE=BASE/'ckpt/ckpt.py'
POLICIES={'mainrun':{'allowed_user':'labmain','cpu_limit':8,'priority':1000},
          'teams':{'allowed_user':'labteam','cpu_limit':1,'priority':10}}

def admit(user,account,cpus):
    if account not in POLICIES:return {'result':'deny','reason':'unknown account'}
    p=POLICIES[account]
    if user!=p['allowed_user']:return {'result':'deny','reason':'association missing'}
    if cpus>p['cpu_limit']:return {'result':'deny','reason':'limit'}
    return {'result':'allow','reason':'simulated policy, not real Slurm'}

def train(root,steps,resume=False,save=False):
    cmd=[sys.executable,str(ENGINE),'train','--directory',str(root),'--steps',str(steps)]
    if resume:cmd.append('--resume')
    if save:cmd.append('--save')
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=75)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out');a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='g7-ERA-v5-sim-') as td:
        td=pathlib.Path(td)
        baseline=train(td/'baseline',12)
        before=train(td/'recover',5,save=True)
        # NODE FAILURE is **represented** by the process terminating at S5;
        # no real hardware is harmed or failed.
        after=train(td/'recover',12,resume=True)
        same_records=(baseline['records']==before['records']+after['records'])
        same_state=(baseline['state_digest']==after['state_digest'])
        same_rng=(baseline['rng_digest']==after['rng_digest'])
        outcomes={'account_admission':{'mainrun_permitted':admit('labmain','mainrun',4),
                    'teams_limit_denied':admit('labteam','teams',2),
                    'mainrun_unauthorized_denied':admit('labteam','mainrun',1)},
                  'checkpoint':{'model_loss_bits_equal':same_records,'state_digest_equal':same_state,
                                'rng_digest_equal':same_rng,'uninterrupted_steps':12,
                                'restart_after_step':5,'resumed_steps':len(after['records'])},
                  'simulated_events':[{'event':'job_admitted','account':'mainrun'},
                                      {'event':'node_lease_lost','at_step':5,'simulated':True},
                                      {'event':'requeue','note':'Slurm not executed; CPU new process invoked'},
                                      {'event':'checkpoint_restored','from_step':5},
                                      {'event':'training_completed','step':12}],
                  'evidence_level':'REAL_CPU_TRAINING_PLUS_SYNTHETIC_SCHEDULER',
                  'not_proven':['Slurm runtime','GPU','NCCL','DeepSpeed engine','S3 remote durability']}
        assert same_records and same_state and same_rng
        assert outcomes['account_admission']['teams_limit_denied']['result']=='deny'
        if a.out:
            out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(outcomes,indent=2)+'\n')
        print(json.dumps({'result':'PASS','scope':outcomes['evidence_level'],'details':outcomes},sort_keys=True))
if __name__=='__main__':main()
