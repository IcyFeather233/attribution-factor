import subprocess
import tarfile

import pytest

from qwen_gpt55_experiments.cleanup_old_extractions import remove_extraction, save_changes


def test_cleanup_does_not_follow_links_outside_target(tmp_path):
    pending = tmp_path / 'pending'
    target = pending / 'old'
    target.mkdir(parents=True)
    outside = tmp_path / 'keep'
    outside.mkdir()
    (outside / 'valuable.txt').write_text('retain')
    (target / 'alias').symlink_to(outside, target_is_directory=True)
    (target / 'file.txt').write_text('old data')
    result = remove_extraction(target, pending)
    assert not target.exists()
    assert (outside / 'valuable.txt').read_text() == 'retain'
    assert result['removed_files'] >= 1
    with pytest.raises(RuntimeError):
        remove_extraction(outside, pending)


def test_cleanup_exports_tracked_and_untracked_source_changes(tmp_path):
    source = tmp_path / 'source'
    work = source / 'uuid/ROOT/testbed'
    work.mkdir(parents=True)
    def git(*args):
        subprocess.run(['git', '-C', str(work), *args], check=True, capture_output=True)
    git('init')
    (work / 'a.py').write_text('old\n')
    git('add', 'a.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
    (work / 'a.py').write_text('new\n')
    (work / 'new_test.py').write_text('test source')
    backup = tmp_path / 'backup'
    save_changes(source, backup)
    patch = (backup / 'uuid/tracked_changes.patch').read_text()
    assert '+new' in patch and '-old' in patch
    with tarfile.open(backup / 'uuid/untracked_files.tar.gz') as archive:
        assert archive.extractfile('new_test.py').read() == b'test source'
