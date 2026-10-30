"""Verify this support release without importing or executing archived research code."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'Delelstm', 'IMV_full', 'IMV_tensor', 'Retain', 'LSTM'}
DATASETS = {'Exchange', 'Electricity', 'PM2.5'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def under(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe path in index')
    path = root / rel
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), 'Symlink in indexed path')
    require(root.resolve() in path.resolve().parents, 'Path outside root')
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def verify(root=ROOT):
    manifest = json.loads((root / 'provenance/release_manifest.json').read_text())
    records = manifest['files']
    names = [r['path'] for r in records]
    require(len(names) == len(set(names)), 'Duplicate manifest paths')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts}
    require(actual == set(names) | {'provenance/release_manifest.json'}, 'Release inventory differs from manifest')
    for item in records:
        p = under(root, item['path'])
        require(p.is_file(), 'Missing file: ' + item['path'])
        require(p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], 'Hash mismatch: ' + item['path'])
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=item['path'])

    raw = rows(root / 'results/formal_results_75_runs.csv')
    summary = rows(root / 'results/model_summary_mean_sampleSD_15_rows.csv')
    expected = {(d, m, e) for d in DATASETS for m in MODELS for e in range(5)}
    identities = [(r['dataset'], r['model'], int(r['experiment_id'])) for r in raw]
    require(len(raw) == 75 and len(set(identities)) == 75 and set(identities) == expected, 'Invalid formal result identities')
    provenance = json.loads((root / 'provenance/result_sources.json').read_text())
    sources = {(r['dataset'], r['model'], r['experiment_id']) for r in provenance}
    require(len(provenance) == 75 and sources == expected, 'Invalid result provenance identities')
    runs = json.loads((root / 'provenance/run_summaries.json').read_text())
    require(len(runs) == 3 and {r['dataset'] for r in runs} == DATASETS, 'Invalid run summaries')
    run_ids = {r['dataset']: r['source_run_id'] for r in runs}
    epoch_limits = {'Exchange': 300, 'Electricity': 350, 'PM2.5': 300}
    for row in raw:
        require(row['source_run_id'] == run_ids[row['dataset']], 'Run identity mismatch')
        require(0 <= int(row['best_validation_epoch_zero_based']) < epoch_limits[row['dataset']], 'Best epoch out of range')
        values = [float(row[k]) for k in ['test_mse', 'test_rmse', 'test_mae', 'test_mape_raw_ratio', 'test_mape_percent']]
        require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid numeric result')
        require(close(float(row['test_rmse']) ** 2, row['test_mse']), 'RMSE/MSE mismatch')
        require(close(float(row['test_mape_raw_ratio']) * 100, row['test_mape_percent']), 'MAPE unit mismatch')
    summary_ids = [(r['dataset'], r['model']) for r in summary]
    require(len(summary) == 15 and len(set(summary_ids)) == 15 and set(summary_ids) == {(d, m) for d in DATASETS for m in MODELS}, 'Invalid summary identities')
    for row in summary:
        group = [r for r in raw if (r['dataset'], r['model']) == (row['dataset'], row['model'])]
        require(int(row['n']) == len(group) == 5, 'Summary count mismatch')
        for metric in ['test_rmse', 'test_mae', 'test_mape_percent']:
            values = [float(r[metric]) for r in group]
            require(close(row[metric + '_mean'], statistics.mean(values)), 'Mean mismatch')
            require(close(row[metric + '_sample_sd'], statistics.stdev(values)), 'Sample SD mismatch')
    require(sum(r['counts']['models_completed'] for r in runs) == 75, 'Model count mismatch')
    require(sum(r['counts']['epochs_completed'] for r in runs) == 23750, 'Epoch count mismatch')
    require(sum(r['counts']['optimizer_step_successes'] for r in runs) == 732500, 'Update count mismatch')
    for run in runs:
        require(run['status'] == 'COMPLETED' and run['exit_code'] == 0, 'Incomplete historical run')
        require(run['overall_numerical_stability'] == 'UNRESOLVED', 'Scientific classification changed')
    pm = next(r for r in runs if r['dataset'] == 'PM2.5')
    require(pm['deviation_label'] == 'Controlled Execution Deviation — Restored Five-model Comparison', 'PM deviation label changed')
    for record in json.loads((root / 'provenance/harness_sources.json').read_text()):
        p = under(root, record['release_path'])
        require(digest(p) == record['sha256'] and p.stat().st_size == record['bytes'], 'Archived-code provenance mismatch')
    return {'files_checked': len(records), 'formal_results': 75, 'summary_rows': 15, 'sample_sd_ddof': 1, 'model_execution': False, 'status': 'PASS'}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), sort_keys=True))
    except (ValueError, OSError, KeyError, SyntaxError) as error:
        print('FAIL: ' + str(error), file=sys.stderr)
        sys.exit(1)
