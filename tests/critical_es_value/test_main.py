import pytest

from critical_es_value import main


@pytest.mark.parametrize(
    "confidence, alternative, expected, match",
    [
        (0.90, "one-sided", 0.10, None),
        (0.95, "one-sided", 0.05, None),
        (0.90, "two-sided", 0.05, None),
        (0.95, "two-sided", 0.025, None),
        (0, "one-sided", 0, r"confidence must be in \(0, 1\)"),
        (1, "one-sided", 0, r"confidence must be in \(0, 1\)"),
        (0.95, "invalid", 0, r"alternative must be one of 'one-sided' or 'two-sided'"),
    ],
)
def test_get_alpha(confidence, alternative, expected, match):
    if match:
        with pytest.raises(ValueError, match=match):
            main.get_alpha(confidence, alternative)
    else:
        alpha = main.get_alpha(confidence, alternative)
        assert alpha == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "dof, expected",
    [
        (10, 0.92274560805),
        (20, 0.96194453374),
        (30, 0.97475437821),
    ],
)
def test_get_J(dof, expected):
    assert main.get_J(dof) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "dataset, alternative, confidence, expected",
    [
        (
            "test_dataset1",
            "two-sided",
            0.95,
            {
                "d": 0.7479307,
                "d_critical": 0.3734061,
                "b_critical": 0.3609507,
                "g": 0.728391,
                "g_critical": 0.3636509,
            },
        ),
        (
            "test_dataset2",
            "two-sided",
            0.95,
            {
                "d": 1.986911,
                "d_critical": 0.2841969,
                "b_critical": 0.3661223,
                "g": 1.956317,
                "g_critical": 0.2798208,
            },
        ),
    ],
)
def test_critical_from_one_sample_ttest(
    test_dataset1, test_dataset2, dataset, alternative, confidence, expected
):
    data = test_dataset1 if dataset == "test_dataset1" else test_dataset2
    result = main.critical_from_one_sample_ttest(
        x=data["x"],
        alternative=alternative,
        confidence=confidence,
    ).iloc[0]
    assert result["d"] == pytest.approx(expected["d"], abs=1e-6)
    assert result["d_critical"] == pytest.approx(expected["d_critical"], abs=1e-6)
    assert result["b_critical"] == pytest.approx(expected["b_critical"], abs=1e-6)
    assert result["g"] == pytest.approx(expected["g"], abs=1e-6)
    assert result["g_critical"] == pytest.approx(expected["g_critical"], abs=1e-6)


@pytest.mark.parametrize(
    "dataset, paired, alternative, correction, confidence, expected",
    [
        (
            "test_dataset1",
            False,
            "two-sided",
            True,
            0.95,
            {
                "d": 0.5072601,
                "d_critical": 0.5168963,
                "b_critical": 0.5183234,
                "g": 0.5006344,
                "g_critical": 0.5101447,
            },
        ),
        (
            "test_dataset2",
            False,
            "two-sided",
            True,
            0.95,
            {
                "d": -0.0993843,
                "d_critical": 0.3975426,
                "b_critical": 0.4389254,
                "g": -0.0985214,
                "g_critical": 0.3940911,
            },
        ),
    ],
)
def test_critical_from_two_sample_ttest(
    test_dataset1,
    test_dataset2,
    dataset,
    paired,
    alternative,
    correction,
    confidence,
    expected,
):
    data = test_dataset1 if dataset == "test_dataset1" else test_dataset2
    result = main.critical_from_two_sample_ttest(
        x=data["x"],
        y=data["y"],
        paired=paired,
        alternative=alternative,
        correction=correction,
        confidence=confidence,
    ).iloc[0]
    assert result["d"] == pytest.approx(expected["d"], abs=1e-6)
    assert result["d_critical"] == pytest.approx(expected["d_critical"], abs=1e-6)
    assert result["b_critical"] == pytest.approx(expected["b_critical"], abs=1e-6)
    assert result["g"] == pytest.approx(expected["g"], abs=1e-6)
    assert result["g_critical"] == pytest.approx(expected["g_critical"], abs=1e-6)
