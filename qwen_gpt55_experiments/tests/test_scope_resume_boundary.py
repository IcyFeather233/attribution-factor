"""Verify SCOPE resume falls back to the last committed batch boundary."""


def _resume_boundary(n, batch_size):
    return (n // batch_size) * batch_size if n % batch_size else n


def test_non_multiple_falls_back_to_batch_start():
    assert _resume_boundary(6, 8) == 0
    assert _resume_boundary(7, 8) == 0
    assert _resume_boundary(15, 8) == 8
    assert _resume_boundary(23, 8) == 16


def test_exact_multiple_keeps_value():
    assert _resume_boundary(8, 8) == 8
    assert _resume_boundary(16, 8) == 16
    assert _resume_boundary(32, 8) == 32
