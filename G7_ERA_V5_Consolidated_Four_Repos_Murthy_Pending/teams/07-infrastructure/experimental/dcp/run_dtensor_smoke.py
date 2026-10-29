#!/usr/bin/env python3
"""Bounded subprocess coordinator; if Gloo unavailable exits failure, never fake success."""
import json,os,pathlib,socket,subprocess,sys,tempfile,time
W=pathlib.Path(__file__).with_name('cpu_dtensor_smoke.py')

def reserve_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def run_group(mode, root, world=2, timeout=60):
    port=reserve_port();base=dict(os.environ,MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),WORLD_SIZE=str(world),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    if sys.platform=='darwin':base.setdefault('GLOO_SOCKET_IFNAME','lo0')
    procs=[]
    try:
        for rank in range(world):
            env=dict(base,RANK=str(rank),LOCAL_RANK=str(rank))
            procs.append(subprocess.Popen([sys.executable,str(W),'--mode',mode,'--root',str(root),'--world-size',str(world)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        results=[];end=time.monotonic()+timeout
        for i,p in enumerate(procs):
            try:o,e=p.communicate(timeout=max(1,end-time.monotonic()))
            except subprocess.TimeoutExpired:raise RuntimeError(f'Gloo timeout mode={mode} rank={i}')
            results.append({'rank':i,'returncode':p.returncode,'stdout':o[-1500:],'stderr':e[-2500:]})
        if any(r['returncode']!=0 for r in results):raise RuntimeError(json.dumps(results))
        return results
    finally:
        for p in procs:
            if p.poll() is None:p.kill()
            try:p.communicate(timeout=3)
            except Exception:pass

def main():
    with tempfile.TemporaryDirectory(prefix='g7-dcp-smoke-') as d:
        run_group('save',d);res=run_group('load',d)
        assert any('"result": "PASS"' in r['stdout'] for r in res)
        print(json.dumps({'status':'PASS','scope':'CPU-only DCP, 2 ranks, separate OS processes','not_proven':['GPU','DeepSpeed ZeRO optimizer','remote durability','topology change']}))
if __name__=='__main__':main()
