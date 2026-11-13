from attribution_experiments.common.evaluator import normalize_patch_transport


def test_patch_transport_adds_only_missing_final_newline():
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"
    assert normalize_patch_transport(patch) == patch + "\n"
    assert normalize_patch_transport(patch + "\n") == patch + "\n"
    assert normalize_patch_transport("") == ""


def test_patch_transport_restores_stripped_blank_context_at_eof():
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -2,3 +2,3 @@\n context\n-old\n+new"
    assert normalize_patch_transport(patch) == patch + "\n \n"
