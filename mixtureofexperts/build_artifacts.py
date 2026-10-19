from pathlib import Path
import json, math, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent; art=ROOT/'artifacts'/'verified'; figs=ROOT/'figures';figs.mkdir(exist_ok=True)
x=json.loads((art/'results.json').read_text()); control=json.loads((art/'controls.json').read_text())['runs'][0]
d=[v for v in x['history'] if v['phase']=='dense'];m=[v for v in x['history'] if v['phase']=='moe']; total=d[-1]['step']
fig,ax=plt.subplots(figsize=(11,5.4)); ax.plot([a['step'] for a in d],[a['val_ce'] for a in d],'-o',label='Dense pretraining',markersize=4)
ax.plot([total+a['step'] for a in m],[a['val_ce'] for a in m],'-o',label='Converted sparse MoE',markersize=4)
ax.plot([total+a['step'] for a in control['history']],[a['val_ce'] for a in control['history']],'--',label='Dense continuation control')
ax.axvline(total,color='gray',ls=':',lw=1.4); ax.text(total+2,5.14,'FFN → top-2 MoE',va='center',fontsize=9)
ax.set(xlabel='Optimizer updates (continuous timeline)',ylabel='Fixed held-out next-byte cross-entropy (nats)',title='Session 14 | Dense training → MoE upcycling (measured CPU experiment)');ax.grid(alpha=.16);ax.legend();fig.tight_layout();fig.savefig(figs/'loss_and_matched_control.png',dpi=170);plt.close(fig)

fig,ax=plt.subplots(figsize=(10,4.8));us=x['routing_on_heldout']; N=len(us[0]); import numpy as np
ind=np.arange(N);width=.33;expected=sum(us[0])/N
for i,counts in enumerate(us):ax.bar(ind+(i-.5)*width,counts,width,label=f'MoE layer {i+1}')
ax.axhline(expected,color='gray',ls='--',label='Balanced expected assignments')
ax.set_xticks(ind,[f'Expert {j}' for j in ind]);ax.set(ylabel='Selections on 6 held-out evaluation batches',title='Every expert receives tokens; load remains uneven');ax.legend();ax.grid(axis='y',alpha=.16);fig.tight_layout();fig.savefig(figs/'expert_routing.png',dpi=170);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.5))
p=[x['config']['param_dense'],x['config']['param_moe_total'],x['config']['param_moe_active_per_token']]
bars=ax.bar(['Dense (all active)','MoE (stored total)','MoE (active/token)'],p)
for bar,v in zip(bars,p):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3500,f'{v:,}',ha='center')
ax.set(ylabel='Trainable parameters',title='Sparse MoE increases total capacity, not necessarily speed');ax.set_ylim(0,300000);ax.grid(axis='y',alpha=.14);fig.tight_layout();fig.savefig(figs/'parameter_accounting.png',dpi=170);plt.close(fig)

with (art/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['phase','step','continuous_step','train_ce','val_ce','aux','tokens_seen','seconds','tokens_per_sec']);w.writeheader()
    for p,rows,off in [('dense',d,0),('moe',m,total),('dense_control',control['history'],total)]:
        for v in rows:w.writerow({key:(off+v['step'] if key=='continuous_step' else p if key=='phase' else v.get(key)) for key in w.fieldnames})
summary={'experiment':'ERA V5 Session 14 dense to sparse MoE upcycling','dataset_sha256':x['config']['dataset']['sha256'],'dense_steps':total,'moe_steps':m[-1]['step'],
         'dense_initial_ce':d[0]['val_ce'],'dense_final_ce':d[-1]['val_ce'],'moe_start_ce':m[0]['val_ce'],'moe_final_ce':m[-1]['val_ce'],
         'dense_control_final_ce':control['final_ce'],'moe_improvement':m[0]['val_ce']-m[-1]['val_ce'],
         'moe_minus_control_ce':m[-1]['val_ce']-control['final_ce'],
         'dense_parameters':x['config']['param_dense'],'moe_total_parameters':x['config']['param_moe_total'],
         'moe_active_parameters':x['config']['param_moe_active_per_token'],
         'conversion_max_abs_logit_difference':x['config']['conversion_max_logit_abs_error'],
         'experts_per_layer':x['config']['experts'],'active_experts_per_token':x['config']['top_k'],
         'heldout_routing_counts':us,
         'note':'Measured CPU float32 on a 300-row educational instruction sample; single seed; no large-scale speed claim.'}
(art/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
