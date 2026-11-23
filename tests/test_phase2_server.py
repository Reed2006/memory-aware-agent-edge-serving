from pathlib import Path

import pytest

from phase2 import server


def test_default_disables_idle_host_cache_and_explicit_flag_enables(tmp_path):
    args = server.parser().parse_args([])
    command = server.build_command(Path('model.gguf'), args, tmp_path)
    assert '--no-cache-idle-slots' in command and '--cache-idle-slots' not in command
    assert command[command.index('--cache-ram') + 1] == '0'
    args = server.parser().parse_args(['--cache-idle-slots', '--cache-ram', '3072'])
    command = server.build_command(Path('model.gguf'), args, tmp_path)
    assert '--cache-idle-slots' in command and '--no-cache-idle-slots' not in command
    assert '--no-context-shift' in command and '--no-kv-unified' in command


def test_precision_and_slot_budget_are_validated():
    with pytest.raises(ValueError):
        server.validate_args(server.parser().parse_args(['--parallel', '0']))
    with pytest.raises(ValueError):
        server.validate_args(server.parser().parse_args(['--ctx-size', '24577']))


def test_model_hash_streams_file_and_record_path_is_explicit(tmp_path):
    import hashlib
    path = tmp_path / 'm.gguf'
    path.write_bytes(b'test-model')
    assert server.file_hash(path) == hashlib.sha256(b'test-model').hexdigest()
    args = server.parser().parse_args(['--server-record', str(tmp_path / 'custom.json')])
    assert Path(args.server_record) == tmp_path / 'custom.json'


def test_launcher_records_exact_flags_hash_and_actual_cache_path(tmp_path, monkeypatch):
    import json
    import httpx
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'fake-model')
    class Process:
        pid = 123
        def poll(self):
            return None
    monkeypatch.setattr(server, 'assert_port_free', lambda port: None)
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *a, **k: Process())
    class PsProcess:
        def create_time(self):
            return 1234.0
    monkeypatch.setattr(server.psutil, 'Process', lambda pid: PsProcess())
    monkeypatch.setattr(server.httpx, 'get', lambda *a, **k: httpx.Response(200, json={'status': 'ok'}))
    record = tmp_path / 'pointer.json'
    args = server.parser().parse_args(['--model', str(model), '--results-dir', str(tmp_path / 'run'),
        '--server-record', str(record), '--cache-idle-slots', '--cache-ram', '3072'])
    info = server.start(args)
    assert json.loads(record.read_text()) == info
    assert info['model_hash'] == server.file_hash(model) and info['kv_precision'] == 'f16'
    assert info['slot_save_path'] == str((tmp_path / 'run/cache').resolve())
    assert info['cache_idle_slots'] and '--cache-idle-slots' in info['command']


def test_stop_validates_process_identity_and_removes_only_pointer(tmp_path, monkeypatch):
    import json
    import psutil
    record = tmp_path / 'server.json'
    server.stop(record)
    record.write_text(json.dumps({'pid': 123, 'process_create_time': 12}))
    calls = []
    class Process:
        def cmdline(self):
            return ['llama-server']
        def create_time(self):
            return 12
        def wait(self, timeout):
            calls.append('wait')
    monkeypatch.setattr(server.psutil, 'Process', lambda pid: Process())
    monkeypatch.setattr(server.os, 'kill', lambda pid, sig: calls.append('term'))
    server.stop(record)
    assert calls == ['term', 'wait'] and not record.exists()
    record.write_text(json.dumps({'pid': 123, 'process_create_time': 13}))
    server.stop(record)
    assert calls == ['term', 'wait'] and not record.exists()


def test_startup_failure_terminates_process_and_clears_pointer(tmp_path, monkeypatch):
    import httpx
    model, record = tmp_path / 'model', tmp_path / 'pointer'
    model.write_bytes(b'mock')
    calls = []
    class Process:
        pid = 123
        def poll(self):
            return 1
    class PsProcess:
        def create_time(self):
            return 12
    monkeypatch.setattr(server.psutil, 'Process', lambda pid: PsProcess())
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *a, **k: Process())
    monkeypatch.setattr(server, 'assert_port_free', lambda p: None)
    args = server.parser().parse_args(['--model', str(model), '--results-dir', str(tmp_path / 'run'),
                                      '--server-record', str(record)])
    with pytest.raises(RuntimeError, match='startup'):
        server.start(args)
    assert not record.exists()
    with pytest.raises(ValueError, match='model file'):
        server.start(server.parser().parse_args(['--model', str(tmp_path / 'absent')]))
