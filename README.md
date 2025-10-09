# critical_es_value

Calculate critical effect size values.

## Usage

```python
import numpy as np
import pingouin as pg
from critical_es_value import critical_from_two_sample_ttest

np.random.seed(123)
mean, cov, n = [4, 5], [(1, .6), (.6, 1)], 30
x, y = np.random.multivariate_normal(mean, cov, n).T

pg.ttest(x, y, paired=False, alternative="two-sided", correction=True, confidence=0.95)
critical_from_two_sample_ttest(x, y, paired=False, alternative="two-sided", correction=True, confidence=0.95)
```
|        |        T |     dof | alternative   |      p-val | CI95%         |   cohen-d |   BF10 |    power |
|:-------|---------:|--------:|:--------------|-----------:|:--------------|----------:|-------:|---------:|
| T-test | -3.40071 | 55.7835 | two-sided     | 0.00124841 | [-1.68 -0.43] |  0.878059 | 26.155 | 0.916807 |


|          |        T |     dof |   T_critical |         d |   d_critical |   b_critical |         g |   g_critical |
|:---------|---------:|--------:|-------------:|----------:|-------------:|-------------:|----------:|-------------:|
| critical | -3.40071 | 55.7835 |      2.00341 | -0.878059 |     0.517279 |     0.621295 | -0.866191 |     0.510288 |


## Acknowlegement

* [R package](https://psicostat.github.io/criticalESvalue/index.html)
* [Original paper](https://journals.sagepub.com/doi/10.1177/25152459251335298?icid=int.sj-full-text.similar-articles.5)
> Perugini, A., Gambarota, F., Toffalini, E., Lakens, D., Pastore, M., Finos, L., ... & Altoè, G. (2025). The Benefits of Reporting Critical-Effect-Size Values. Advances in Methods and Practices in Psychological Science, 8(2), 25152459251335298.

