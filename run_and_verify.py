"""Execute the raw Colab notebook top-to-bottom and write measured outputs.

Usage: python run_and_verify.py
This is a local reproducibility runner, NOT a replacement for Colab execution.
"""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).parent.resolve()
input_path=ROOT/'ERA_V5_Session_09_Loss_Harness.ipynb'
nb=nbformat.read(input_path,as_version=4)
out=NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
output_path=ROOT/'ERA_V5_Session_09_EXECUTED.ipynb'
nbformat.write(out,output_path)
assert (ROOT/'results.json').exists()
print('PASS:',len(out.cells),'cells executed; wrote',output_path.name,'and results.json')
