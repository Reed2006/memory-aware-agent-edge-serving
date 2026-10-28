from phase1.startup import parse_log, stamp_s

LOG = """\
0.00.077.340 I srv  llama_server: initializing ...
0.00.079.247 I srv    load_model: loading model '/models/Qwen3-4B-Q4_K_M.gguf'
0.00.963.333 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 24576, kv_unified = 'true'
0.00.970.494 I srv  llama_server: model loaded
0.00.970.500 I srv  llama_server: listening on http://127.0.0.1:18080
0.03.896.552 I slot print_timing: id  0 | task 0 |       total time =     511.46 ms /    21 tokens
1.02.000.000 I slot print_timing: id  1 | task 9 |       total time =    9000.00 ms /  5000 tokens
"""


def test_timestamps_are_minutes_seconds_milliseconds_microseconds():
    assert abs(stamp_s('1.02.500.250 I srv  x') - 62.50025) < 1e-9
    assert stamp_s('no timestamp here') is None


def test_launch_row_has_load_ready_and_only_the_first_request(tmp_path):
    path = tmp_path / 'server-P4-C24576-1793029441.log'
    path.write_text(LOG)
    row = parse_log(path)
    assert (row['P'], row['ctx_size'], row['launched_unix']) == (4, 24576, 1793029441)
    assert row['kv_unified'] is True and row['ctx_per_slot'] == 24576
    assert abs(row['load_s'] - (0.970494 - 0.079247)) < 1e-9
    assert abs(row['ready_s'] - 0.9705) < 1e-9
    assert (row['first_request_ms'], row['first_request_tokens']) == (511.46, 21)


def test_launch_that_served_no_request_has_no_first_request(tmp_path):
    path = tmp_path / 'server-P2-C24576-1.log'
    path.write_text('\n'.join(LOG.splitlines()[:5]))
    row = parse_log(path)
    assert row['load_s'] is not None and row['first_request_ms'] is None
