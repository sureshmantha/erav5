"""Continuation controls: same pretrained dense checkpoint, identical held-out evaluation,
identical continuation data order, same optimizer resets, router regularization sweep.
"""
import argparse, copy, json
from pathlib import Path
import torch
from experiment import setup,dataset,batches,TinyLM,convert,train,SEED

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='artifacts/main');p.add_argument('--data',default='data/era_v5_clean_300.jsonl')
    p.add_argument('--coeffs',nargs='*',type=float,default=[0.1]);p.add_argument('--steps',type=int,default=180)
    a=p.parse_args(); root=Path(a.out)
    x=json.loads((root/'results.json').read_text()); c=x['config']; cfg=c['d']
    setup(c['seed'])
    docs,_=dataset(a.data,c['context']);val=batches(docs['val'],c['context'],c['batch'],6,7042)
    dense=TinyLM(d=cfg['d'],w=cfg['ffn'],layers=cfg['layers'],heads=cfg['heads'],ctx=cfg['ctx'])
    ck=torch.load(root/'dense_last.pt',weights_only=True,map_location='cpu')['model'];dense.load_state_dict(ck)
    initial_moe=convert(dense,c['experts'],c['top_k'])
    models=[('dense_continuation',dense,None)]+[(f'moe_aux_{coeff:g}',copy.deepcopy(initial_moe),coeff) for coeff in a.coeffs]
    out=[]
    for name,model,coefficient in models:
        phase='moe' if coefficient is not None else 'dense'
        print('RUN',name,flush=True)
        h,_=train(model,docs['train'],val,torch.device('cpu'),a.steps,c['batch'],c['context'],
                  c['seed']+18,c['lr'],coefficient or 0.0,phase,15)
        usage=[]
        if coefficient is not None:
            with torch.inference_mode():
                model.eval()
                for xb,yb in val:
                    _=model(xb)
                    counts=[blk.ffn.last_counts for blk in model.blocks]
                    if not usage:usage=[[0]*c['experts'] for _ in counts]
                    for lay,row in enumerate(counts):
                        for e,v in enumerate(row): usage[lay][e]+=v
        out.append({'name':name,'coefficient':coefficient,'initial_ce':h[0]['val_ce'],
                    'final_ce':h[-1]['val_ce'],'usage':usage,'history':h})
    (root/'controls.json').write_text(json.dumps({'runs':out,'comparison':'Controls share checkpoint, batches, eval, LR, steps, optimizer reset.'},indent=2))
    for item in out:print('CONTROL_RESULT',item['name'],item['initial_ce'],item['final_ce'],item['usage'],flush=True)
