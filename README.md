# critical_es_value

Calculate critical effect size values.

## Usage

```python
import numpy as np
import pingouin as pg
from critical_es_value import (
    critical_for_one_sample_ttest,
    critical_for_two_sample_ttest,
    critical_for_correlation_test,
)

np.random.seed(123)
mean, cov, n = [4, 5], [(1, .6), (.6, 1)], 30
x, y = np.random.multivariate_normal(mean, cov, n).T
```

### t-Test


```python
pg.ttest(x, 0)
critical_for_one_sample_ttest(x)
```

|        |       T |   dof | alternative   |       p-val | CI95%       |   cohen-d |      BF10 |   power |
|:-------|--------:|------:|:--------------|------------:|:------------|----------:|----------:|--------:|
| T-test | 16.0765 |    29 | two-sided     | 5.54732e-16 | [3.37 4.35] |   2.93515 | 1.031e+13 |     nan |

|          |       T |   dof |   T_critical |       d |   d_critical |   b_critical |       g |   g_critical |
|:---------|--------:|------:|-------------:|--------:|-------------:|-------------:|--------:|-------------:|
| critical | 16.0765 |    29 |      2.04523 | 2.93515 |     0.373406 |     0.491162 | 2.85847 |     0.363651 |

```python
pg.ttest(x, y)
critical_for_two_sample_ttest(x, y)
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
critical_for_correlation_test(x, y)
```

|         |   n |        r | CI95%       |      p-val |   BF10 |    power |
|:--------|----:|---------:|:------------|-----------:|-------:|---------:|
| pearson |  30 | 0.594785 | [0.3  0.79] | 0.00052736 | 69.723 | 0.950373 |

|          |   n |        r |   dof |   r_critical |    se_r |   se_r_critical |
|:---------|----:|---------:|------:|-------------:|--------:|----------------:|
| critical |  30 | 0.594785 |    28 |     0.361007 | 0.15192 |        0.176238 |

## Acknowlegement

* [R package](https://psicostat.github.io/criticalESvalue/index.html)
* [Original paper](https://journals.sagepub.com/doi/10.1177/25152459251335298?icid=int.sj-full-text.similar-articles.5)
> Perugini, A., Gambarota, F., Toffalini, E., Lakens, D., Pastore, M., Finos, L., ... & Altoè, G. (2025). The Benefits of Reporting Critical-Effect-Size Values. Advances in Methods and Practices in Psychological Science, 8(2), 25152459251335298.

