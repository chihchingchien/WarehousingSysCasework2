from pathlib import Path
import zipfile,hashlib,json
b=Path(__file__).resolve().parent
source=Path(json.loads((b/'outputs/data_audit.json').read_text())['source'])
a=json.loads((b/'outputs/data_audit.json').read_text())
a['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
(b/'outputs/data_audit.json').write_text(json.dumps(a,indent=2))
with zipfile.ZipFile(b/'Task1_v2_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(b.rglob('*')):
        if f.is_file() and f.suffix!='.zip' and '__pycache__' not in f.parts and not f.name.startswith('.'):
            z.write(f,Path('hyperbasket_task1_v2')/f.relative_to(b))
print(b/'Task1_v2_complete.zip')
