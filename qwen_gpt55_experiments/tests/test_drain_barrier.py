import fcntl
import os

import pytest

from qwen_gpt55_experiments.drain_remote_slots import install_barriers


def test_drain_keeps_running_holder_and_blocks_new_admissions(tmp_path):
    slots = tmp_path / 'slots'
    slots.mkdir()
    path = slots / 'slot-000.lock'
    with path.open('w+') as legacy:
        fcntl.flock(legacy.fileno(), fcntl.LOCK_EX)
        old_inode = os.fstat(legacy.fileno()).st_ino
        archive = tmp_path / 'old_inodes'
        barriers = install_barriers(slots, archive, 1)
        try:
            assert (archive / path.name).stat().st_ino == old_inode
            assert path.stat().st_ino != old_inode
            # The old task still owns its original slot until it finishes.
            with (archive / path.name).open() as original:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(legacy, fcntl.LOCK_UN)
            # Finishing a task must not let a queued legacy worker reacquire it.
            with path.open() as newcomer:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(newcomer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (archive / path.name).open() as original:
                fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            for handle in barriers:
                handle.close()
    with path.open() as resumed:
        fcntl.flock(resumed, fcntl.LOCK_EX | fcntl.LOCK_NB)
