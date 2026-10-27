"""Package code and the final Phase 1 results for handoff (e.g. to AWS); exclude credentials and model weights."""

import hashlib
import json
import zipfile
from pathlib import Path


def safe_files(root: Path, secret: bytes | None):
    roots = ['README.md', '.gitignore', 'requirements-lock.txt',
             'phase1', 'tests', 'scripts', 'docs', 'runs']
    for name in roots:
        selected = root / name
        candidates = [selected] if selected.is_file() else sorted(selected.rglob('*'))
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if ('__pycache__' in relative.parts or '.pytest_cache' in relative.parts
                    or (relative.parts[0] == 'runs' and 'cache' in relative.parts)
                    or path.name in ['API.md', '.DS_Store'] or path.suffix in ['.gguf', '.bin']):
                continue
            if secret and (secret in path.read_bytes() or secret in str(relative).encode()):
                raise ValueError('Credential detected in a deliverable; archive creation aborted.')
            yield path


def main():
    root = Path.cwd()
    for required in ['runs/final/PHASE1_REPORT.md', 'runs/final/restore_validation.json', 'runs/final/manifest.json']:
        if not (root / required).exists():
            raise SystemExit(f'Required completed artifact missing: {required}')
    secret_path = root / 'API.md'
    secret = secret_path.read_bytes().strip() if secret_path.exists() else None
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ['phase1', 'tests', 'scripts']
                     for p in sorted((root / folder).rglob('*'))
                     if p.is_file() and '__pycache__' not in p.parts}
    (root / 'runs/final/source-hashes.json').write_text(json.dumps(source_hashes, indent=2))
    target = root / 'artifacts/phase1-local-results.zip'
    target.parent.mkdir(exist_ok=True)
    files = list(safe_files(root, secret))
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
        license_path = root / 'third_party/tau2-bench/LICENSE'
        if license_path.exists():
            archive.write(license_path, 'third_party_notices/tau2-bench.LICENSE')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(Path(name).name == 'API.md' or name.endswith('.gguf') for name in archive.namelist())
        if secret:
            assert all(secret not in archive.read(name) for name in archive.namelist())
    with target.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    target.with_suffix('.zip.sha256').write_text(checksum + '  ' + target.name + '\n')
    print(f'Created {target.name}: {target.stat().st_size/2**20:.2f} MiB, {len(files)} project files; key and weights excluded.')


if __name__ == '__main__':
    main()
