"""打包当前报告与可复现实验代码，排除原始行情、归档、权重和逐笔大文件。"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/submission.zip'
files=set()
for name in ['README.md','requirements.txt','Makefile','.gitignore','code.py','code2.py',
             'code2_new.py','code2_research.py','code2_trading_data.py','code2_xgboost.py',
             'code2_tuning.py','code2_pipeline.py','output/pdf/report.pdf']:
    files.add(ROOT/name)
for directory in ['report','tests','tools','audit']:
    files.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for path in (ROOT/'results').rglob('*'):
    if not path.is_file():continue
    if path.suffix not in ('.csv','.json','.log'):continue
    if path.name in ('daily_ledger.csv','daily_ledger_gross.csv','gross_ledger.csv'):continue
    if path.parent.name=='models':continue  # 月度权重；保留其父目录的配置和调参记录。
    files.add(path)
config=json.loads((ROOT/'results/run_config.json').read_text())
for name,digest in config['sources'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
    for path in sorted(files):archive.write(path,str(path.relative_to(ROOT)))
    archive.writestr('SHA256SUMS.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(OUT) as archive:
    assert archive.testzip() is None
    for name,digest in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
print(f'{OUT.name}: {len(files)} files, {OUT.stat().st_size:,} bytes; all hashes verified')
