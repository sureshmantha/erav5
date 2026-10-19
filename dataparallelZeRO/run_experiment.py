"""Reproduce notebook results and export plots, CSV and JSON in repository."""
import json,sys,os,csv,platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import virtual_zero as z

def produce(base=ROOT):
    torch.set_num_threads(1)
    (base/'figures').mkdir(exist_ok=True,parents=True)
    (base/'results').mkdir(exist_ok=True,parents=True)
    trials=z.validate()
    compact=[]
    for t in trials:
        compact.append({k:v for k,v in t.items() if k not in ('final','rank_state_elements')})
    toy=z.memory_table()
    large=z.projected_state_for_30b()
    output=dict(seed_weights=20260912,seed_data=312,world_size=32,real_gpu_count=0,
        virtual_ranks=32,model='2 linear layer classifier with ReLU (synthetic teacher labels)',
        model_parameter_count=z.NUM_PARAMS,global_batch=64,batch_per_rank=2,
        optimizer='Adam beta1=.9 beta2=.999 eps=1e-8 LR=.008, fp64 execution',
        runtime_torch=torch.__version__,python=platform.python_version(),cpu_threads=1,
        experimental_results=compact,ideal_mixed_precision_toy=toy,
        analytical_projection_30b=large,
        estimated_Z3_minimum_one_gathered_layer_bytes=z.stage3_extra_buffer_bytes(),
        notes=['No GPUs used; local CPU torch.cat models all-gather semantics without networking.',
          'Communication estimates are ideal ring-collective payload per rank, not measured bandwidth/time.',
          'Ideal mixed-precision memory is separate from actual fp64 CPU allocations.',
          'ZeRO3 extra gathered layer and activation allocations are excluded from the persistent-state table.',
          'Forward/backward dense model FLOPs per rank are unchanged, Adam state updates per rank decrease.'])
    (base/'results'/'experiment.json').write_text(json.dumps(output,indent=2))
    with (base/'results'/'memory_and_communication.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['stage','bytes_per_param_per_rank','30b_state_GiB','30b_ring_GB_per_rank','ring_units_of_fp16_param_bytes', 'optimizer_elements_per_rank_avg'])
        writer.writeheader()
        for s,t in enumerate(large):
            writer.writerow(dict(stage=s,bytes_per_param_per_rank=z.resident_bytes_per_parameter(s),
                **{'30b_state_GiB':round(t['resident_bytes_per_rank']/2**30,5),
                '30b_ring_GB_per_rank':round(t['modeled_ring_bytes_per_rank']/1e9,5),
                'ring_units_of_fp16_param_bytes':t['communication_S_units'],
                'optimizer_elements_per_rank_avg':30e9/(1 if s==0 else 32)}))
    colors=['#1f3855','#217da9','#e7ab37','#58b8a6']
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    x=np.arange(4)
    value=[t['resident_bytes_per_rank']/2**30 for t in large]
    bars=axs[0].bar(x,value,color=colors)
    axs[0].set_xticks(x,[f'ZeRO-{s}' for s in x]);axs[0].set_ylabel('Persistent state per virtual GPU (GiB)')
    axs[0].set_title('30B-parameter example · 32 ranks')
    for b,v in zip(bars,value):axs[0].text(b.get_x()+b.get_width()/2,v+4,f'{v:.1f}',ha='center',fontsize=10)
    axs[0].set_ylim(0,max(value)*1.15)
    val=[t['modeled_ring_bytes_per_rank']/1e9 for t in large]
    bars=axs[1].bar(x,val,color=colors)
    axs[1].set_xticks(x,[f'ZeRO-{s}' for s in x]);axs[1].set_ylabel('Ideal ring payload / rank / step (GB)')
    axs[1].set_title('Illustrative communication volume · NOT timing')
    for b,v in zip(bars,val):axs[1].text(b.get_x()+b.get_width()/2,v+3,f'{v:.2f}',ha='center',fontsize=10)
    axs[1].set_ylim(0,max(val)*1.15)
    fig.suptitle('ZeRO trades persistent state for communication',fontsize=15,fontweight='bold')
    fig.tight_layout();fig.savefig(base/'figures'/'state_vs_communication.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.8,4.4))
    for t,c in zip(compact,colors):
        ax.plot(np.arange(1,len(t['losses'])+1),t['losses'],marker='o',color=c,
                alpha=.85,linewidth=2,label=f"ZeRO-{t['stage']}")
    ax.set_title('Real PyTorch training · 32 logical ranks · four identical trajectories')
    ax.set_xlabel('Optimizer step');ax.set_ylabel('Global supervised cross-entropy (pre-update)')
    ax.set_xticks(range(1,len(compact[0]['losses'])+1));ax.grid(alpha=.2);ax.legend(ncol=4,loc='upper right')
    fig.tight_layout();fig.savefig(base/'figures'/'loss_equivalence.png',dpi=170,bbox_inches='tight');plt.close(fig)

    # Exact rank-level state distribution; derived from resident tensors and byte assumptions.
    per_rank=[]
    for stage in range(4):
        row=[]
        for r in trials[stage]['rank_state_elements']:
            # BF16 W/G are 2 bytes each; 12B optimizer = FP32 master+first+second moment.
            # There are TWO stored moments in code; ideal accounting includes separate FP32 master.
            o=(r['moments']//2)*12
            row.append((r['weight']*2+r['gradient']*2+o)/1024)
        per_rank.append(row)
    fig,ax=plt.subplots(figsize=(12,3.8))
    im=ax.imshow(per_rank,aspect='auto',cmap='YlGnBu')
    ax.set_yticks(range(4),[f'ZeRO-{j}' for j in range(4)])
    ax.set_xticks(range(0,32,2),[f'{j}' for j in range(0,32,2)])
    ax.set_xlabel('Virtual rank (0–31)')
    ax.set_title('32 ranks · resident training-state distribution (ideal KB)')
    fig.colorbar(im,ax=ax,label='KiB / rank',shrink=.8)
    fig.tight_layout();fig.savefig(base/'figures'/'32_rank_ownership.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    b1=ax.bar(np.arange(4)-0.18,[1,1,1,1],.36,color='#217da9',label='Dense fwd+bwd FLOPs per rank (relative)')
    b2=ax.bar(np.arange(4)+0.18,[1,1/32,1/32,1/32],.36,color='#e7ab37',label='Optimizer element updates per rank (relative)')
    ax.set_xticks(range(4),[f'ZeRO-{s}' for s in range(4)])
    ax.set_ylim(0,1.25);ax.set_title('Sharding optimizer work ≠ skipping model computation')
    ax.legend(fontsize=9);ax.set_ylabel('Relative to replicated data parallel baseline')
    fig.tight_layout();fig.savefig(base/'figures'/'compute_vs_optimizer.png',dpi=170,bbox_inches='tight');plt.close(fig)

    print('Validation: 32 ranks, ZeRO-0/1/2/3; all tests passed')
    print('The training losses are identical to displayed precision:')
    for t in compact:
        print(f"  stage {t['stage']}: losses={[round(x,9) for x in t['losses']]} "+
           f"max_weight_error={t['max_parameter_error_vs_baseline']:.3g}, "+
           f"mean CPU step={t['mean_cpu_step_ms']:.2f} ms")
    print('Analytical persistent-state example, 30B parameters / 32 ranks:')
    for s,t in enumerate(large):
        print(f"  stage {s}: {z.resident_bytes_per_parameter(s):.4f} bytes/param/rank, "+
            f"{t['resident_bytes_per_rank']/2**30:.3f} GiB, ring {t['modeled_ring_bytes_per_rank']/1e9:.3f} GB/rank/step")
    print('Important: GPU time, actual GPU allocated VRAM and actual network timings are NOT measured.')
    return output

if __name__=='__main__':produce()
