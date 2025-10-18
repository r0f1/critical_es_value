# critical_es_value

Calculate critical effect size values for t-Tests, correlation tests and linear regression coefficients.

## Installation

```
pip install critical-es-value
```

## Overview

This package offers 8 main functions. Either provide the raw sample data to the main version of the function, or provide the summary statistics to the `from_values()` version.

| Main version    | from_values()    |
|:----|:----|
| `critical_for_one_sample_ttest()` | `critical_for_one_sample_ttest_from_values()` |
| `critical_for_two_sample_ttest()` | `critical_for_two_sample_ttest_from_values()` |
| `critical_for_correlation_test()` | `critical_for_correlation_test_from_values()` |
| `critical_for_linear_regression()` | `critical_for_linear_regression_from_values()` |


## Main Usage

```python
import numpy as np
import pingouin as pg
import critical_es_value as cev

np.random.seed(123)
mean, cov, n = [4, 5], [(1, .6), (.6, 1)], 30
x, y = np.random.multivariate_normal(mean, cov, n).T
```

### t-Test

```python
pg.ttest(x, 0)
cev.critical_for_one_sample_ttest(x)
```

|        |       T |   dof | alternative   |       p-val | CI95%       |   cohen-d |      BF10 |   power |
|:-------|--------:|------:|:--------------|------------:|:------------|----------:|----------:|--------:|
| T-test | 16.0765 |    29 | two-sided     | 5.54732e-16 | [3.37 4.35] |   2.93515 | 1.031e+13 |     nan |

|          |       T |   dof |   T_critical |       d |   d_critical |   b_critical |       g |   g_critical |
|:---------|--------:|------:|-------------:|--------:|-------------:|-------------:|--------:|-------------:|
| critical | 16.0765 |    29 |      2.04523 | 2.93515 |     0.373406 |     0.491162 | 2.85847 |     0.363651 |

```python
pg.ttest(x, y, paired=False)
cev.critical_for_two_sample_ttest(x, y, paired=False)
```

|        |        T |   dof | alternative   |     p-val | CI95%         |   cohen-d |   BF10 |    power |
|:-------|---------:|------:|:--------------|----------:|:--------------|----------:|-------:|---------:|
| T-test | -3.40071 |    58 | two-sided     | 0.0012224 | [-1.68 -0.43] |  0.878059 | 26.155 | 0.916807 |


|          |        T |   dof |   T_critical |         d |   d_critical |   b_critical |         g |   g_critical |
|:---------|---------:|------:|-------------:|----------:|-------------:|-------------:|----------:|-------------:|
| critical | -3.40071 |    58 |      2.00172 | -0.878059 |     0.516841 |      0.62077 | -0.866647 |     0.510124 |


### Correlation Test

```python
pg.corr(x, y)
cev.critical_for_correlation_test(x, y)
```

|         |   n |        r | CI95%       |      p-val |   BF10 |    power |
|:--------|----:|---------:|:------------|-----------:|-------:|---------:|
| pearson |  30 | 0.594785 | [0.3  0.79] | 0.00052736 | 69.723 | 0.950373 |

|          |   n |        r |   dof |   r_critical |    se_r |   se_r_critical |
|:---------|----:|---------:|------:|-------------:|--------:|----------------:|
| critical |  30 | 0.594785 |    28 |     0.361007 | 0.15192 |        0.176238 |


### Linear Regression

```python
import pandas as pd

np.random.seed(123)
data = pd.DataFrame({"X": x, "Y": y, "Z": np.random.normal(5, 1, 30)})

pg.linear_regression(data[["X", "Z"]], data["Y"])
cev.critical_for_linear_regression(data[["X", "Z"]], data["Y"])
```

|    | names     |       coef |       se |         T |        pval |       r2 |   adj_r2 |   CI[2.5%] |   CI[97.5%] |
|---:|:----------|-----------:|---------:|----------:|------------:|---------:|---------:|-----------:|------------:|
|  0 | Intercept |  3.15799   | 0.844129 |  3.74112  | 0.000874245 | 0.354522 | 0.306709 |   1.42598  |    4.88999  |
|  1 | X         |  0.487772  | 0.126736 |  3.84871  | 0.000659501 | 0.354522 | 0.306709 |   0.22773  |    0.747814 |
|  2 | Z         | -0.0249309 | 0.140417 | -0.177548 | 0.860403    | 0.354522 | 0.306709 |  -0.313044 |    0.263182 |

|    | names     |       coef |   coef_critical |
|---:|:----------|-----------:|----------------:|
|  0 | Intercept |  3.15799   |        1.73201  |
|  1 | X         |  0.487772  |        0.260042 |
|  2 | Z         | -0.0249309 |        0.288113 |


## Usage of from_values() version

### t-Test

```python
t_test_result = pg.ttest(x, 0).iloc[0]
cev.critical_for_one_sample_ttest_from_values(
    t=t_test_result["T"],
    n=len(x),
    dof=t_test_result.dof,
    std=np.std(x, ddof=1),
)
```

|          |       T |   dof |   T_critical |       d |   d_critical |       g |   g_critical |   b_critical |
|:---------|--------:|------:|-------------:|--------:|-------------:|--------:|-------------:|-------------:|
| critical | 16.0765 |    29 |      2.04523 | 2.93515 |     0.373406 | 2.85847 |     0.363651 |     0.491162 |

```python
t_test_result = pg.ttest(x, y, paired=False).iloc[0]
cev.critical_for_two_sample_ttest_from_values(
    t=t_test_result["T"],
    n1=len(x),
    n2=len(y),
    dof=t_test_result.dof,
    paired=False,
    std1=np.std(x, ddof=1),
    std2=np.std(y, ddof=1),
)
```

|          |        T |   dof |   T_critical |         d |   d_critical |         g |   g_critical |   b_critical |
|:---------|---------:|------:|-------------:|----------:|-------------:|----------:|-------------:|-------------:|
| critical | -3.40071 |    58 |      2.00172 | -0.878059 |     0.516841 | -0.866647 |     0.510124 |      0.62077 |


### Correlation Test

```python
corr_test = pg.corr(x, y).iloc[0]
cev.critical_for_correlation_test_from_values(
    r=corr_test["r"],
    n=corr_test["n"],
    variant="ttest",
)
```

|          |   n |        r |   dof |   r_critical |    se_r |   se_r_critical |
|:---------|----:|---------:|------:|-------------:|--------:|----------------:|
| critical |  30 | 0.594785 |    28 |     0.361007 | 0.15192 |        0.176238 |


### Linear Regression

```python
model = pg.linear_regression(data[["X", "Z"]], data["Y"])

cev.critical_for_linear_regression_from_values(
    coeffs=model["coef"].values,
    coeffs_se=model["se"].values,
    coeffs_names=model["names"].values,
    dof=model.df_resid_,
    variant="ttest",
)
```

|    | names     |       coef |   coef_critical |
|---:|:----------|-----------:|----------------:|
|  0 | Intercept |  3.15799   |        1.73201  |
|  1 | X         |  0.487772  |        0.260042 |
|  2 | Z         | -0.0249309 |        0.288113 |


## Resources

* [R package](https://github.com/psicostat/criticalESvalue)
* [Original paper](https://journals.sagepub.com/doi/10.1177/25152459251335298?icid=int.sj-full-text.similar-articles.5)

 > Perugini, A., Gambarota, F., Toffalini, E., Lakens, D., Pastore, M., Finos, L., ... & Altoè, G. (2025). The Benefits of Reporting Critical-Effect-Size Values. Advances in Methods and Practices in Psychological Science, 8(2), 25152459251335298.
