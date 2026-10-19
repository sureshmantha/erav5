"""Build fully self-contained Colab notebook. No GitHub/raw URL dependency."""
from pathlib import Path
import base64,gzip
import nbformat as nbf
root=Path(__file__).resolve().parent
ex=(root/'src/experiment.py').read_text();con=(root/'src/controls.py').read_text()
plots=(root/'build_artifacts.py').read_text();tests=(root/'tests/test_experiment.py').read_text()
data=base64.b64encode(gzip.compress((root/'data/era_v5_clean_300.jsonl').read_bytes(),compresslevel=9)).decode()
nb=nbf.v4.new_notebook()
nb['metadata']={'colab':{'name':'ERA V5 — Session 14 Dense-to-MoE (reproducible)','provenance':[]},
                'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}}
M=nbf.v4.new_markdown_cell; C=nbf.v4.new_code_cell
nb.cells=[
M('''# ERA V5 — Session 14 | Dense FFN → Sparse Mixture of Experts

**Assignment:** Train a dense/linear feed-forward language model, convert the learned FFNs to routed sparse MoE, and prove training continues with **decreasing loss**.

The model is small by design; it is a real byte-level causal next-token Transformer. The independent validation set is document-disjoint, the initial untrained loss is sanity-checked against `ln(257)`, and the conversion starts from a pretrained dense checkpoint. This notebook is **self-contained**: it embeds the exact 300-record prior ERA V5 cleaned teaching subset, the source code, tests, and plots. **Research/teaching data only; upstream noncommercial restrictions apply.**

Three full experiments run below: 60 updates dense, 90 updates MoE, and 90 updates matched dense continuation. Exact results from the prior CPU run are in the repository `artifacts/verified/`.
'''),
C('''from pathlib import Path
import importlib.util, subprocess, sys
missing=[pkg for pkg in ('torch','numpy','matplotlib','pytest') if importlib.util.find_spec(pkg) is None]
if missing: subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
import platform, torch, numpy, matplotlib, pytest
print('Python:',platform.python_version(),'PyTorch:',torch.__version__)
print('CUDA:',torch.cuda.is_available(),'GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('Next steps: reconstruct deterministic educational fixture, execute correctness tests, train models.')'''),
C("""from pathlib import Path
Path('src').mkdir(exist_ok=True)
Path('tests').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
Path('artifacts/verified').mkdir(parents=True,exist_ok=True)
Path('src/experiment.py').write_text("""+repr(ex)+""")
Path('src/controls.py').write_text("""+repr(con)+""")
Path('build_artifacts.py').write_text("""+repr(plots)+""")
Path('tests/test_experiment.py').write_text("""+repr(tests)+""")
import gzip, base64, hashlib
raw=gzip.decompress(base64.b64decode("""+repr(data)+"""))
assert hashlib.sha256(raw).hexdigest()=='41f311e4db037900aac621b1850221c15d1c5431c49a0b14fdb554ca7b8b8e9c'
Path('data/era_v5_clean_300.jsonl').write_bytes(raw)
print('Deterministic sample restored, SHA256:',hashlib.sha256(raw).hexdigest())
print('All source code and training records restored locally.')"""),
M('''## Part A — Correctness gate

Before running a longer experiment, inspect the **actual shapes**, check that future tokens do not leak through attention, confirm **next-token shift** without crossing document boundaries, and validate MoE conversion and selected-only gradients. The test suite also asserts that the initial CE is close to the expected uniform-vocabulary CE. The final test uses the results file if it already exists.'''),
C('''import subprocess,sys
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_experiment.py'],check=True)'''),
M('''## Part B — Train the dense model, upcycle FFN weights, continue with MoE

The MoE has six FFN experts per transformer block. Only two run per token. **All six expert FFNs start as exact copies** of the pretrained dense FFN. Because the two selected router weights sum to one, the conversion preserves the function before any new optimizer steps. The auxiliary balance loss has coefficient `0.01`. Results are measured, not prefilled.'''),
C('''subprocess.run([sys.executable,'-u','src/experiment.py',
    '--dense-steps','60','--moe-steps','90','--eval-every','10',
    '--out','artifacts/verified'],check=True)'''),
M('''## Part C — The fair dense-continuation control

Reset optimizer state for both arms, reuse exactly the same dense checkpoint, the same 90 additional minibatch samples, and the same fixed validation windows. A lower MoE CE is **not automatically evidence of superior modeling**, particularly if the numerical gap is tiny.'''),
C('''subprocess.run([sys.executable,'-u','src/controls.py','--out','artifacts/verified','--coeffs','--steps','90'],check=True)'''),
M('''## Part D — Measured curves, expert coverage and parameter audit

Dense, MoE and dense continuation appear together on the same optimizer-update axis. The routing chart reveals uneven load rather than assuming perfect balance.'''),
C('''subprocess.run([sys.executable,'build_artifacts.py'],check=True)
from IPython.display import display,Image
for filename in ('loss_and_matched_control.png','expert_routing.png','parameter_accounting.png'):
    print(filename);display(Image(filename='figures/'+filename))'''),
C('''import json, math
summary=json.loads(Path('artifacts/verified/summary.json').read_text())
assert summary['moe_final_ce'] < summary['moe_start_ce']
assert summary['dense_final_ce'] < summary['dense_initial_ce']
print('DENSE initial/final:',summary['dense_initial_ce'],summary['dense_final_ce'])
print('MOE at conversion/final:',summary['moe_start_ce'],summary['moe_final_ce'])
print('Dense continuation:',summary['dense_control_final_ce'])
print('Conversion maximum absolute logit difference:',summary['conversion_max_abs_logit_difference'])
print('All routed experts active on evaluation:',all(min(v)>0 for v in summary['heldout_routing_counts']))
print('Initial CE vs theoretical log(257):',summary['dense_initial_ce'],math.log(257))'''),
M('''## Oral defense — explain it without an agent

1. Why can cloning a trained FFN into all routed experts preserve logits *exactly in algebra* even when the router scores differ?
2. Why can the total MoE parameter count rise while the per-token active count is less than its stored total? Why might the MoE still run slower than dense?
3. What happens to unselected expert gradients, and why does the router receive any gradients when top-k is discrete?
4. Does our result prove experts specialized in topics? (No.) Does the small difference against dense continuation prove MoE wins? (No.)
5. Which part came from Sessions 9–13 (loss shift, gradients, optimizer reset, ZeRO vs computation, reversible activation memory)?

**Submission:** keep the actual executed notebook outputs, source files, test results, fixed evaluation protocol and README. A live GitHub repository URL must be added *only after publishing*.
''')]
path=root/'ERA_V5_S14_Dense_to_MoE.ipynb';nbf.write(nb,path)
print('Notebook created',path,'size',path.stat().st_size,'bytes',len(nb.cells),'cells')
