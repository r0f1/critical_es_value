import pytest

from critical_es_value import main

@pytest.mark.parametrize(
    "confidence, alternative, expected",
    [
        (0.95, "one-sided", 0.05),
    ])
def test_get_alpha(confidence, alternative, expected):

    alpha = main.get_alpha(confidence, alternative)
    assert alpha == pytest.approx(expected, abs=1e-6)
