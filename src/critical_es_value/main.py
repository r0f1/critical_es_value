import numpy as np
import pandas as pd
import pingouin
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import loggamma


def get_alpha(confidence: float, alternative: str) -> float:
    """Calculate the significance level (alpha) corresponding to a given confidence level.

    Args:
        confidence (float): Confidence level between 0 and 1 (exclusive).
        alternative (str): The alternative hypothesis. Either "one-sided" or "two-sided".

    Returns:
        float: The significance level (alpha).

    Raises:
        ValueError: If `confidence` is not in (0, 1)
        ValueError: If `alternative` is not one of "one-sided" or "two-sided".

    Examples:
        >>> get_alpha(0.95, "one-sided")
        0.05
        >>> get_alpha(0.95, "two-sided")
        0.025
    """

    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be in (0, 1)")
    if alternative not in ("one-sided", "two-sided"):
        raise ValueError("alternative must be one of 'one-sided' or 'two-sided'")

    alpha = 1 - confidence

    if alternative == "two-sided":
        return alpha / 2
    return alpha


def get_J(dof: int) -> np.float64:
    """Calculate the bias correction factor J for Hedges' g.

    Args:
        dof (int): Degrees of freedom.

    Returns:
        np.float64: The bias correction factor J.

    Examples:
        >>> get_J(10)
        0.92274560805
        >>> get_J(20)
        0.96194453374
    """
    num = loggamma(dof / 2)
    denom = np.log(np.sqrt(dof / 2)) + loggamma((dof - 1) / 2)
    return np.exp(num - denom)


def critical_from_one_sample_ttest(
    x: ArrayLike,
    alternative: str,
    confidence: float,
) -> pd.DataFrame:
    """Calculate critical effect size values from a one-sample t-test.

    Returns a DataFrame with the following columns:
     - T: t-value of the test statistic
     - dof: Degrees of freedom
     - T_critical: Critical t-value
     - d: Cohen's d
     - d_critical: Critical value for Cohen's d
     - b_critical: Critical value for the raw mean difference
     - g: Hedges' g
     - g_critical: Critical value for Hedges' g

    Args:
        x (array-like): Sample data.
        alternative (str): The alternative hypothesis. Either "one-sided" or "two-sided".
        confidence (float): Confidence level between 0 and 1 (exclusive).

    Returns:
        pd.DataFrame: A DataFrame containing critical effect size values.
    """

    t_test_result = pingouin.ttest(
        x=x,
        y=0,
        paired=False,
        alternative=alternative,
        correction=False,
        confidence=confidence,
    ).iloc[0]

    alpha = get_alpha(confidence, alternative)
    dof = t_test_result.dof

    n = len(x)
    factor = np.sqrt(1 / n)

    t = t_test_result["T"]
    d = t * factor

    tc = np.abs(stats.t.ppf(alpha, dof))
    dc = tc * factor

    j = get_J(dof)

    return pd.DataFrame(
        [
            {
                "T": t,
                "dof": dof,
                "T_critical": tc,
                "d": d,
                "d_critical": dc,
                "b_critical": tc * np.std(x, ddof=1) / np.sqrt(n),
                "g": d * j,
                "g_critical": dc * j,
            }
        ],
        index=["critical"],
    )


def _critical_from_two_sample_ttest_paired(
    x: ArrayLike,
    y: ArrayLike,
    alternative: str,
    correction: bool,
    confidence: float,
) -> pd.DataFrame:
    """Calculate critical effect size values from a PAIRED two-sample t-test.

    Returns a DataFrame with the following columns:
     - T: t-value of the test statistic
     - dof: Degrees of freedom
     - T_critical: Critical t-value
     - d: Cohen's d
     - d_critical: Critical value for Cohen's d
     - b_critical: Critical value for the raw mean difference
     - g: Hedges' g
     - g_critical: Critical value for Hedges' g
     - dz: Cohen's dz
     - dz_critical: Critical value for Cohen's dz
     - gz: Hedges' gz
     - gz_critical: Critical value for Hedges' gz

    Args:
        x (array-like): Sample data for group 1.
        y (array-like): Sample data for group 2.
        alternative (str): The alternative hypothesis. Either "one-sided" or "two-sided".
        correction (bool): Whether to apply Welch's correction for unequal variances.
        confidence (float): Confidence level between 0 and 1 (exclusive).

    Returns:
        pd.DataFrame: A DataFrame containing critical effect size values.
    """

    if len(x) != len(y):
        raise ValueError("For paired tests, x and y must have the same length.")

    t_test_result = pingouin.ttest(
        x=x,
        y=y,
        paired=True,
        alternative=alternative,
        correction=correction,
        confidence=confidence,
    ).iloc[0]

    alpha = get_alpha(confidence, alternative)
    dof = t_test_result.dof
    n = len(x)

    r12 = np.corrcoef(x, y)[0, 1]
    factor1 = np.sqrt(1 / n)
    factor2 = np.sqrt(2 * (1 - r12))

    t = t_test_result["T"]
    dz = t * factor1
    d = dz * factor2

    tc = np.abs(stats.t.ppf(alpha, dof))
    dzc = tc * factor1
    dc = dzc * factor2

    j = get_J(dof)

    return pd.DataFrame(
        [
            {
                "T": t,
                "dof": dof,
                "T_critical": tc,
                "d": d,
                "d_critical": dc,
                "b_critical": tc * np.std(x - y, ddof=1) / np.sqrt(n),
                "g": d * j,
                "g_critical": dc * j,
                "dz": dz,
                "dz_critical": dzc,
                "gz": dz * j,
                "gz_critical": dzc * j,
            }
        ],
        index=["critical"],
    )


def critical_from_two_sample_ttest(
    x: ArrayLike,
    y: ArrayLike,
    paired: bool,
    alternative: str,
    correction: bool,
    confidence: float,
) -> pd.DataFrame:
    """Calculate critical effect size values from a paired or unpaired two-sample t-test.

    Returns a DataFrame with the following columns:
     - T: t-value of the test statistic
     - dof: Degrees of freedom
     - T_critical: Critical t-value
     - d: Cohen's d
     - d_critical: Critical value for Cohen's d
     - b_critical: Critical value for the raw mean difference
     - g: Hedges' g
     - g_critical: Critical value for Hedges' g

    Args:
        x (array-like): Sample data for group 1.
        y (array-like): Sample data for group 2.
        paired (bool): Whether the samples are paired.
        alternative (str): The alternative hypothesis. Either "one-sided" or "two-sided".
        correction (bool): Whether to apply Welch's correction for unequal variances.
        confidence (float): Confidence level between 0 and 1 (exclusive).

    Returns:
        pd.DataFrame: A DataFrame containing critical effect size values.
    """

    if paired:
        return _critical_from_two_sample_ttest_paired(
            x=x,
            y=y,
            alternative=alternative,
            correction=correction,
            confidence=confidence,
        )

    t_test_result = pingouin.ttest(
        x=x,
        y=y,
        paired=paired,
        alternative=alternative,
        correction=correction,
        confidence=confidence,
    ).iloc[0]

    alpha = get_alpha(confidence, alternative)
    dof = t_test_result.dof

    n1 = len(x)
    n2 = len(y)

    factor = np.sqrt(1 / n1 + 1 / n2)

    t = t_test_result["T"]
    d = t * factor

    tc = np.abs(stats.t.ppf(alpha, dof))
    dc = tc * factor

    s1 = np.std(x, ddof=1)
    s2 = np.std(y, ddof=1)

    if correction:
        se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    else:
        se = np.sqrt((s1**2 * (n1 - 1) + s2**2 * (n2 - 1)) / (n1 + n2 - 2)) * factor

    j = get_J(dof)

    return pd.DataFrame(
        [
            {
                "T": t,
                "dof": dof,
                "T_critical": tc,
                "d": d,
                "d_critical": dc,
                "b_critical": tc * se,
                "g": d * j,
                "g_critical": dc * j,
            }
        ],
        index=["critical"],
    )
