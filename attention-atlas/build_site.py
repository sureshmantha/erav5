from pathlib import Path
p=Path(__file__).parent
s=(p/'index.template.html').read_text(encoding='utf-8')
s=s.replace('__CSS__',(p/'styles.css').read_text(encoding='utf-8')).replace('__JS__',(p/'app.js').read_text(encoding='utf-8')).replace('__DATA__',(p/'mechanisms.json').read_text(encoding='utf-8').strip())
(p/'index.html').write_text(s,encoding='utf-8')
print('Built standalone index.html:',len(s),'characters')
