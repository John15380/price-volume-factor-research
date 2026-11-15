"""打包可审阅的报告、代码、汇总表和日志；大体积行情与交易明细留在工作区。"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output/report_code2_definitions_bundle.zip'
paths = set()
for name in ['code2.py', 'code2_new.py', 'code2_research.py', 'code2_xgboost.py',
             'code2_trading_data.py', 'code2_new_说明.md', 'output/pdf/report_code2_definitions.pdf',
             'audit/code2_check_results.json', 'audit/full_a_price_factors.py',
             'audit/constraints_sensitivity.py', 'audit/研报逐项核对.md',
             'audit/recheck_factor_definitions.py','audit/因子定义复核.md',
             'report/figures/char.png', 'report/figures/logo.png']:
    paths.add(ROOT / name)
paths.update((ROOT / 'tests').glob('test_code2*.py'))
paths.update(p for p in (ROOT / 'report_definitions').rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and not ('source_reports' in p.parts and p.suffix.lower()=='.pdf'))
paths.add(ROOT/'audit/source_reports/manifest.json')
for directory in ['results_code2_definitions', 'results_research_all_a_constraints','results_code2_definition_audit']:
    parent = ROOT / directory
    for pattern in ['*summary.csv', 'sensitivity.csv', '*config.json', '*diagnostics.json',
                    'limitations.json', '*limits.csv', 'feature_coverage.csv',
                    'coverage.csv','numeric_reference_checks.json',
                    'comparison/summary.csv', 'xgboost/*/training_log.csv',
                    'xgboost/*/feature_importance.csv', 'icir/*/weights.csv']:
        paths.update(parent.glob(pattern))
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
    for p in sorted(paths):
        archive.write(p, str(p.relative_to(ROOT)))
    archive.writestr('SHA256SUMS.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
with zipfile.ZipFile(OUT) as archive:
    assert archive.testzip() is None
    for name, digest in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
print(f'{OUT.name}: {len(paths)} files, {OUT.stat().st_size:,} bytes; all hashes checked')
