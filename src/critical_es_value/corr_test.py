import numpy as np
import pandas as pd
from scipy import stats

from critical_es_value import utils


def critical_for_correlation_using_ttest(
    r: float,
    n: int,
    confidence: float = 0.95,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    alpha = utils.get_alpha(confidence, alternative)
    dof = n - 2

    tc = stats.t.ppf(alpha, dof)
    rc = np.sqrt(tc**2 / (tc**2 + dof))

    return pd.DataFrame(
        [
            {
                "r": r,
                "n": n,
                "dof": dof,
                "r_critical": rc,
                "se_r": np.sqrt((1 - r**2) / dof),
                "se_r_critical": np.sqrt((1 - rc**2) / dof),
            }
        ],
        index=["critical"],
    )


def critical_for_correlation_using_ztest(
    r: float,
    n: int,
    confidence: float = 0.95,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    alpha = utils.get_alpha(confidence, alternative)
    dof = n - 2

    zc = stats.norm.ppf(alpha)
    rc = np.tanh(zc / np.sqrt(n - 3))

    return pd.DataFrame(
        [
            {
                "r": r,
                "n": n,
                "dof": dof,
                "r_critical": rc,
                "se_r": np.sqrt((1 - r**2) / dof),
                "se_r_critical": np.sqrt((1 - rc**2) / dof),
                "rz_critical": np.atanh(rc),
                "se_rz_critical": 1 / np.sqrt(n - 3),
            }
        ],
        index=["critical"],
    )
