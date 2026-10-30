"""Repair-source overlay; preserve original evidence and refresh every checksum."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='output/submission_bundle.zip')
    parser.add_argument('--output', default='output/submission_bundle_fixed.zip')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base, output = Path(args.base).resolve(), Path(args.output).resolve()
    if base == output or output.exists():
        raise FileExistsError('Choose a new output archive; originals are never overwritten')
    with zipfile.ZipFile(base) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt original archive')
        files = {n: archive.read(n) for n in archive.namelist() if not n.endswith('/')}
    for line in files['SHA256SUMS.txt'].decode('utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError(f'Original checksum mismatch: {name}')
    protected = {k:v for k,v in files.items() if k.startswith(('runs/', 'weights/', 'data/', 'output/', 'paper/'))}
    for folder in ('cutout_lab', 'scripts', 'tests', 'repair_validation'):
        for path in (root / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pt', '.zip'):
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    files['README.md'] = (root / 'README.md').read_bytes()
    files['提交说明.md'] = '本包是本地复现入口修复版。按 README 的已有权重复核流程操作；原始论文、权重和证据原样保留。新增核验见 repair_validation/。\n'.encode('utf-8')
    files.pop('SHA256SUMS.txt', None)
    for name, content in protected.items():
        if files[name] != content:
            raise ValueError(f'Original evidence changed: {name}')
    files['SHA256SUMS.txt'] = ('\n'.join(f'{hashlib.sha256(v).hexdigest()}  {k}' for k,v in sorted(files.items())) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(files.items()):
            archive.writestr(name, value)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'output':str(output),'files':len(files),'original_evidence_files_preserved':len(protected),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()
