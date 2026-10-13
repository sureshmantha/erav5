import re,json,urllib.parse
from pathlib import Path
p=Path(__file__).parent;d=json.loads((p/'mechanisms.json').read_text());h=(p/'index.html').read_text()
assert len(d)>=20
assert [x['date'] for x in d]==sorted(x['date'] for x in d)
assert len(set(x['id'] for x in d))==len(d)
for x in d:
 assert all(x.get(k) for k in ('date','name','tag','problem','idea','buy','give','choose','source','basis'))
 assert re.fullmatch(r'\d{4}-\d{2}-\d{2}',x['date'])
 assert urllib.parse.urlparse(x['source']).scheme=='https'
for k in ('Scaled dot-product attention','DeepSeek Sparse Attention','DroPE','Gated DeltaNet','GQA','YaRN','MLA','ALiBi'):
 assert any(k.lower() in x['name'].lower() for x in d),k
assert 'window.ATTENTION_DATA=' in h and 'function renderCost' in h
assert len(h)>40000
print(f'PASS: {len(d)} chronological, source-linked, complete entries and built HTML')