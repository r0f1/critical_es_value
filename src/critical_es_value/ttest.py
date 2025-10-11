import numpy as np
import pandas as pd
import pingouin
from numpy.typing import ArrayLike
from scipy import stats

from critical_es_value import utils


def critical_from_one_sample_ttest(
    x: ArrayLike,
    alternative: str = "two-sided",
    confidence: float = 0.95,
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
        alternative (str): The alternative hypothesis. Either "one-sided" or "two-sided". Default is "two-sided".
        confidence (float): Confidence level between 0 and 1 (exclusive). Default is 0.95.

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

    alpha = utils.get_alpha(confidence, alternative)
    dof = t_test_result.dof

    n = len(x)
    factor = np.sqrt(1 / n)

    t = t_test_result["T"]
    d = t * factor

    tc = np.abs(stats.t.ppf(alpha, dof))
    dc = tc * factor

    j = utils.get_bias_correction_factor_J(dof)

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
    alternative: str = "two-sided",
    correction: bool = False,
    confidence: float = 0.95,
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
        alternative (str): The alternative hypothesis. Either "one-sided" or "two-sided". Default is "two-sided".
        correction (bool): Whether to apply Welch's correction for unequal variances. Default is False.
        confidence (float): Confidence level between 0 and 1 (exclusive). Default is 0.95.

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

    alpha = utils.get_alpha(confidence, alternative)
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

    j = utils.get_bias_correction_factor_J(dof)

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
    paired: bool = False,
    alternative: str = "two-sided",
    correction: bool = False,
    confidence: float = 0.95,
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
        paired (bool): Whether the samples are paired. Default is False.
        alternative (str): The alternative hypothesis. Either "one-sided" or "two-sided". Default is "two-sided".
        correction (bool): Whether to apply Welch's correction for unequal variances. Default is False.
        confidence (float): Confidence level between 0 and 1 (exclusive). Default is 0.95.

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

    alpha = utils.get_alpha(confidence, alternative)
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

    j = utils.get_bias_correction_factor_J(dof)

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
