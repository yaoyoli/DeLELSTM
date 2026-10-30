"""Compare separately held evidence by reading bytes; never load a model/checkpoint."""
import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from verify_release import ROOT, digest, require, under, verify


def audit(source, runtime, archives=False):
    verify()
    index = json.loads((ROOT / 'provenance/frozen_source_index.json').read_text())
    for item in index['files']:
        path = under(source, item['path'])
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'Frozen-source mismatch: ' + item['path'])
    for item in json.loads((ROOT / 'provenance/harness_sources.json').read_text()):
        path = under(runtime, item['runtime_relative_path'])
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'Original Harness mismatch: ' + item['runtime_relative_path'])
    for name in ['requirements-macos-arm64-cp39.lock.txt', 'installed_versions.json', 'verification_status.json']:
        require(digest(under(runtime, 'locks/' + name)) == digest(ROOT / 'environment' / name), 'Environment record mismatch: ' + name)
    sources = json.loads((ROOT / 'provenance/result_sources.json').read_text())
    with (ROOT / 'results/formal_results_75_runs.csv').open(newline='') as stream:
        results = {(r['dataset'], r['model'], int(r['experiment_id'])): r for r in csv.DictReader(stream)}
    for item in sources:
        path = under(runtime, item['runtime_relative_path'])
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'Test CSV mismatch: ' + item['runtime_relative_path'])
        table = list(csv.DictReader(io.StringIO(path.read_text())))
        require(len(table) == 1, 'Unexpected original CSV row count')
        row = results[(item['dataset'], item['model'], item['experiment_id'])]
        for metric in ['mse', 'rmse', 'mae', 'mape']:
            column = 'test_mape_raw_ratio' if metric == 'mape' else 'test_' + metric
            require(float(table[0][metric + '_test']) == float(row[column]), 'Original metric mismatch')
        checkpoint = under(runtime, item['checkpoint_runtime_relative_path'])
        require(checkpoint.stat().st_size == item['checkpoint_bytes'] and digest(checkpoint) == item['checkpoint_sha256'], 'Checkpoint byte hash mismatch')
    if archives:
        for item in json.loads((ROOT / 'provenance/evidence_archives.json').read_text()):
            path = under(runtime, item['runtime_relative_path'])
            require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'Evidence archive mismatch')
            with zipfile.ZipFile(path) as zipped:
                for record in [r for r in sources if r['evidence_archive'] == item['runtime_relative_path']]:
                    require(zipped.read(record['archive_member']) == under(runtime, record['runtime_relative_path']).read_bytes(), 'Archive CSV member mismatch')
    return {'status': 'PASS', 'frozen_files': len(index['files']), 'test_csvs': 75, 'checkpoint_hashes': 75, 'archives_checked': archives, 'model_execution': False, 'checkpoint_deserialization': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--archives', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.source.resolve(), args.runtime.resolve(), args.archives), sort_keys=True))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        print('FAIL: ' + str(error), file=sys.stderr)
        sys.exit(1)
