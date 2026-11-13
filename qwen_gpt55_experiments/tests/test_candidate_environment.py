import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minisweagent.exceptions import Submitted
from qwen_gpt55_experiments.harnessfix.candidate_environment import candidate_environment_class


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = Path(__file__).parent / "fixtures/harnessfix_candidate"
ORIGINAL = ROOT / "qwen_failure_comparison/mini-swe-agent-v2.4.6"
PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"


def environment(tmp_path, source, outputs):
    class Runtime:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(**kwargs, volumes={})
            self.container_name = kwargs["container_name"]
            self.scratch_volumes = {str(tmp_path / name): mount for name, mount in
                                    [("tmp", "/tmp"), ("var_tmp", "/var/tmp"), ("shm", "/dev/shm")]}
            self.calls = []
            self.closed = False

        def _run_host(self, args, **kwargs):
            self.calls.append((args, kwargs))
            output = outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return subprocess.CompletedProcess(args, 0, output, None)

        def cleanup(self):
            self.closed = True

    cls = candidate_environment_class(source, Runtime)
    return cls(repo=tmp_path, executable="udocker", container_name="private-test",
               image="test-image", cwd="/testbed", env={"TEST_FLAG": "yes"})


def test_original_candidate_still_uses_original_submission_semantics(tmp_path):
    env = environment(tmp_path, ORIGINAL, [SENTINEL + "not a diff"])
    with pytest.raises(Submitted):
        env.execute({"command": "submit"})


def test_actual_repair_recovers_invalid_submission_through_private_transport(tmp_path):
    env = environment(tmp_path, CANDIDATE, [SENTINEL + "not a diff", PATCH])
    runtime = env._runtime
    with pytest.raises(Submitted) as caught:
        env.execute({"command": "submit"}, cwd="/testbed/subdir")
    assert caught.value.messages[0]["extra"]["submission_source"] == "auto_git_diff"
    assert caught.value.messages[0]["extra"]["submission"] == PATCH
    assert len(runtime.calls) == 2
    assert runtime.calls[1][0][-1] == "git diff --no-color --"
    for args, _ in runtime.calls:
        assert "--workdir=/testbed/subdir" in args
        for mount in ["/tmp", "/var/tmp", "/dev/shm"]:
            assert any(arg.startswith("--volume=/volume/med/") and arg.endswith(":" + mount) for arg in args)
    assert "--env=TEST_FLAG=yes" in runtime.calls[0][0]
    assert env.serialize()["info"]["config"]["container_transport"] == "persistent_udocker_candidate_v1"
    env.cleanup()
    env.cleanup()
    assert runtime.closed


def test_actual_repair_rejects_invalid_submission_without_a_patch(tmp_path):
    env = environment(tmp_path, CANDIDATE, [SENTINEL + "not a diff", ""])
    result = env.execute({"command": "submit"})
    assert result["returncode"] == 1
    assert result["extra"]["submission_rejected"] is True


def test_actual_repair_accepts_valid_patch_without_extra_command(tmp_path):
    env = environment(tmp_path, CANDIDATE, [SENTINEL + PATCH])
    with pytest.raises(Submitted):
        env.execute({"command": "submit"})
    assert len(env._runtime.calls) == 1


def test_candidate_handles_transport_timeout(tmp_path):
    env = environment(tmp_path, CANDIDATE, [subprocess.TimeoutExpired("run", 10, output=b"partial")])
    result = env.execute({"command": "sleep 20"}, timeout=10)
    assert result["returncode"] == -1
    assert result["output"] == "partial"
    assert env._runtime.calls[0][1]["timeout"] == 10
