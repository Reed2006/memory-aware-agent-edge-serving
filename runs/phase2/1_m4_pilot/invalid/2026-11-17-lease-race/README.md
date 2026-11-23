# Invalid dev runs: lease race (2026-11-17)

These runs used `phase2/scheduler.py` before the fix recorded in
`campaign/lock-history.jsonl` at 2026-11-17 ~06:10.

Bug: when a model step finished, the dispatcher handed the freed slot to the next
queued request before the replay harness called `tool_start()`. The owner therefore
looked idle, and `_protect` returned without pricing a lease, so lease/TTL policies
evicted it unconditionally. In `cache/S101-T0`, 19 of 23 evictions hit an owner in tool
wait, while only 2 lease decisions were made.

Affected policies: cache, joint-*, joint-checkpoint, ttl-*. Policies that never lease
(fixed, fifo, affinity, affinity-fixed, queue, attained) behave identically before
and after the fix, so their completed dev runs remain valid.

- `cache/S101-T0`: completed run, excluded and rerun.
- `joint-a20/S101-T0`: aborted by the operator once the bug was found; rerun.

Regression test: `tests/test_phase2_scheduler.py::test_slot_is_protected_from_the_moment_its_step_ends`.
