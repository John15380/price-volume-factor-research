"""将历史产物可逆归档，校验大小及关键文件哈希；不删除原始行情。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'archive/2026-11-14'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def move(path,destination):
    path=ROOT/path;destination=ROOT/destination
    if not path.exists():return None
    if destination.exists():raise FileExistsError(destination)
    files=[p for p in path.rglob('*') if p.is_file()] if path.is_dir() else [path]
    size=sum(p.stat().st_size for p in files)
    important={str(p.relative_to(path)) if path.is_dir() else p.name:digest(p) for p in files
               if p.suffix in ('.py','.md','.tex','.sty','.cls','.json') or p.name in ('summary.csv','report.pdf')}
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(path),str(destination))
    after=[p for p in destination.rglob('*') if p.is_file()] if destination.is_dir() else [destination]
    assert len(after)==len(files) and sum(p.stat().st_size for p in after)==size
    for relative,sha in important.items():
        current=destination/relative if destination.is_dir() else destination
        assert digest(current)==sha,current
    return {'source':str(path.relative_to(ROOT)),'destination':str(destination.relative_to(ROOT)),
            'files':len(files),'bytes':size,'key_file_sha256':important}


def run():
    if (ROOT/'audit/archive_manifest.json').exists():
        print('历史产物已经归档；未执行任何迁移。')
        return
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    template=ROOT/'tmp/original_report_template';template.mkdir(parents=True,exist_ok=True)
    for name in ['CjC.cls','thesis.sty']:
        shutil.copy2(ROOT/'report'/name,template/name)
    (template/'figures').mkdir(exist_ok=True)
    for name in ['char.png','logo.png']:
        shutil.copy2(ROOT/'report/figures'/name,template/'figures'/name)
    records=[]
    names=[p.name for p in ROOT.glob('results_*') if p.is_dir()]
    names+=['report_new','report_research','report_definitions','output','audit','code2_new_说明.md']
    for name in names:
        record=move(name,Path('archive/2026-11-14')/name)
        if record:records.append(record)
    for name in ['report','report.pdf','plots','plots2','temp_factors_CSI500','temp_processed_features',
                 'code-ori.py','code-0.ipynb','check.ipynb']:
        record=move(name,Path('archive/original')/name)
        if record:records.append(record)
    for folder in ['report/figures','report/tables','output/pdf','audit','results/logs']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    for name in ['CjC.cls','thesis.sty']:
        shutil.copy2(template/name,ROOT/'report'/name)
    for name in ['char.png','logo.png']:
        shutil.copy2(template/'figures'/name,ROOT/'report/figures'/name)
    (ROOT/'audit/archive_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    (ROOT/'archive/README.md').write_text('历史产物已按来源目录归档，没有删除原始行情或历史报告。\n'
        '2026-11-14/为此前生成的报告、实验和审计；original/为原报告模板及原作业辅助产物。\n'
        '当前唯一结果为根目录results/，唯一当前报告源为report/，交付文件为output/。\n'
        '完整迁移记录见audit/archive_manifest.json；Git保留此前代码、报告和汇总。\n')
    print(f'Archived {len(records)} entries; {sum(r["bytes"] for r in records):,} bytes preserved')


if __name__=='__main__':run()
