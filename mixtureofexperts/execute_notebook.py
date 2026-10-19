from pathlib import Path
import nbformat
from nbclient import NotebookClient
r=Path(__file__).resolve().parent
src=r/'ERA_V5_S14_Dense_to_MoE.ipynb';out=r/'ERA_V5_S14_EXECUTED.ipynb'
nb=nbformat.read(src,as_version=4)
client=NotebookClient(nb,timeout=360,kernel_name='python3',resources={'metadata':{'path':str(r)}})
client.execute()
nbformat.write(nb,out)
print('EXECUTION_SUCCESS',out,flush=True)
