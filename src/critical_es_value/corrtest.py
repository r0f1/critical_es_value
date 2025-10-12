import numpy as np
import pandas as pd
import pingouin
from numpy.typing import ArrayLike
from scipy import stats

from critical_es_value import utils


def critical_for_correlation_test(
    x: ArrayLike,
    y: ArrayLike,
    confidence: float = 0.95,
    alternative: str = "two-sided",
    variant: str = "ttest",
) -> pd.DataFrame:
    if variant not in ["ttest", "ztest"]:
        raise ValueError("variant must be one of 'ttest' or 'ztest'")

    corr_test_result = pingouin.corr(
        x, y, alternative=alternative, method="pearson"
    ).iloc[0]

    r = corr_test_result["r"]
    n = corr_test_result["n"]

    alpha = utils.get_alpha(confidence, alternative)
    dof = n - 2

    if variant == "ttest":
        tc = np.abs(stats.t.ppf(alpha, dof))
        rc = np.sqrt(tc**2 / (tc**2 + dof))
    else:
        zc = np.abs(stats.norm.ppf(alpha))
        rc = np.tanh(zc / np.sqrt(n - 3))

    result = {
        "r": r,
        "n": n,
        "dof": dof,
        "r_critical": rc,
        "se_r": np.sqrt((1 - r**2) / dof),
        "se_r_critical": np.sqrt((1 - rc**2) / dof),
    }
    if variant == "ztest":
        result["rz_critical"] = np.atanh(rc)
        result["se_rz_critical"] = 1 / np.sqrt(n - 3)

    return pd.DataFrame([result], index=["critical"])
