from pathlib import Path

import pytest

from phase1.package import safe_files


def test_archive_allowlist_excludes_credentials_weights_and_cache(tmp_path):
    for name in ['README.md', 'API.md', 'models/model.gguf', 'phase1/core.py',
                 'runs/final/cache/state.bin', 'runs/final/raw/request.jsonl']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('example')
    selected = {str(p.relative_to(tmp_path)) for p in safe_files(tmp_path, b'fake-secret')}
    assert selected == {'README.md', 'phase1/core.py', 'runs/final/raw/request.jsonl'}


def test_accidental_secret_in_an_allowed_file_aborts_without_echoing_it(tmp_path):
    secret = b'unique-fake-secret-for-test'
    (tmp_path / 'README.md').write_bytes(secret)
    with pytest.raises(ValueError) as error:
        list(safe_files(tmp_path, secret))
    assert secret.decode() not in str(error.value)
