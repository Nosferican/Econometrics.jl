# Estimators

## The Basics

Regression analysis assumes a model for the data generation process of some outcome. For example, the [Mincer earnings function](https://doi.org/10.1086/258055) assumes that earnings are a function of schooling and experience.

```math
\begin{equation}
\begin{aligned}
  \ln(\text{wage}) &= \beta_{0} + \beta_{1}\text{ schooling} + \beta_{2}\text{ experience} + \beta_{3}\text{ (experience)}^{2} + \varepsilon\\\\
  \varepsilon &\sim \mathcal{N}\left(\mu, \sigma^{2}\right)
\end{aligned}
\end{equation}
```

The response variable, earnings, is usually log-normal distributed (Clementi and Gallegati [2005](https://doi.org/10.1007/88-470-0389-X_1)). In addition, earnings are non-negative and assuming the sample has positive earnings a useful transformation is to take the natural log-normal distributed response. Schooling is commonly measured in years of schooling and experience as the number of years in the labor market. The model assumes a polynomial relationship between years of experience and the log of earnings of degree two.

How could we use Econometrics.jl to analyze such a model?

We can use the RDatasets package for accessing various example datasets.

The following line loads packages for the tutorial.

```julia
using CSV, RDatasets, Econometrics
```




We can load the PSID dataset from the Ecdat R package.

```julia
data = RDatasets.dataset("Ecdat", "PSID")
data = data[data.Earnings .> 0 .&
            data.Kids .< 98,:] # Only those with earnings and valid number of kids
```

```
3652×8 DataFrame
  Row │ IntNum  PersNum  Age    Educatn  Earnings  Hours  Kids   Married
      │ Int32   Int32    Int32  Int32?   Int32     Int32  Int32  Cat…
──────┼────────────────────────────────────────────────────────────────────
    1 │      4        4     39       12     77250   2940      2  married
    2 │      4        6     35       12     12000   2040      2  divorced
    3 │      4        7     33       12      8000    693      1  married
    4 │      4      173     39       10     15000   1904      2  married
    5 │      5        2     47        9      6500   1683      5  married
    6 │      6        4     44       12      6500   2024      2  married
    7 │      6      172     38       16      7000   1144      3  married
    8 │      7        4     38        9      5000   2080      4  divorced
  ⋮   │   ⋮        ⋮       ⋮       ⋮        ⋮        ⋮      ⋮        ⋮
 3646 │   9285        4     41        8      1800    410     98  divorced
 3647 │   9286        2     45       12     20080   2040      2  divorced
 3648 │   9292        2     41        6     13000   2448      5  married
 3649 │   9297        2     42        2      3000   1040      4  married
 3650 │   9302        1     37        8     22045   2793     98  divorced
 3651 │   9305        2     40        6       134     30      3  married
 3652 │   9306        2     37       17     33000   2423      4  married
                                                          3637 rows omitted
```





Baseline model

```julia
model = fit(EconometricModel, # Indicates the default model
            @formula(log(Earnings) ~ Educatn + Age + Age^2), # formula
            data)
```

```
Continuous Response Model
Number of observations: 3652
Null Loglikelihood: -5696.23
Loglikelihood: -5671.17
R-squared: 0.0136
LR Test: 50.11 ∼ χ²(3) ⟹  Pr > χ² = 0.0000
Formula: log(Earnings) ~ 1 + Educatn + Age + (Age ^ 2)
Variance Covariance Estimator: OIM
────────────────────────────────────────────────────────────────────────────────────
                    PE           SE      t-value  Pr > |t|        2.50%       97.50%
────────────────────────────────────────────────────────────────────────────────────
(Intercept)   7.14834      0.957333      7.46693    <1e-12   5.27138     9.0253
Educatn       0.00370285   0.00111874    3.30984    0.0009   0.00150944  0.00589627
Age           0.0934537    0.0493985     1.89183    0.0586  -0.00339765  0.190305
Age ^ 2      -0.000918383  0.000628258  -1.46179    0.1439  -0.00215015  0.000313389
────────────────────────────────────────────────────────────────────────────────────
```





### Frequency weights

We can also specify observation weights,

```julia
model = fit(EconometricModel,
            @formula(log(Earnings) ~ Educatn + Age + Age^2),
            data,
            wts = :PersNum, # frequency weights
            )
```

```
Continuous Response Model
Number of observations: 223174
Null Loglikelihood: -6746726.41
Loglikelihood: -349853.81
R-squared: 0.0064
LR Test: 12793745.20 ∼ χ²(3) ⟹  Pr > χ² = 0.0000
Formula: log(Earnings) ~ 1 + Educatn + Age + (Age ^ 2)
Variance Covariance Estimator: OIM
────────────────────────────────────────────────────────────────────────────────────
                    PE           SE      t-value  Pr > |t|         2.50%      97.50%
────────────────────────────────────────────────────────────────────────────────────
(Intercept)   8.50901      0.131474     64.7201     <1e-99   8.25132      8.7667
Educatn       0.00117042   0.000140001   8.36006    <1e-16   0.000896017  0.00144481
Age           0.0305687    0.00686287    4.45422    <1e-05   0.0171177    0.0440198
Age ^ 2      -0.000167677  8.8448e-5    -1.89576    0.0580  -0.000341032  5.67929e-6
────────────────────────────────────────────────────────────────────────────────────
```





### Categorical variables

Categorical variables can be passed through a contrast or if the feature is coded as a categorical variables it will be handled as categorical.

```julia
model = fit(EconometricModel,
            @formula(log(Earnings) ~ Educatn + Age + Age^2 + Kids),
            data,
            wts = :PersNum,
            contrasts = Dict(:Kids => DummyCoding()))
```

```
Continuous Response Model
Number of observations: 223174
Null Loglikelihood: -6746726.41
Loglikelihood: -345253.18
R-squared: 0.0466
LR Test: 12802946.46 ∼ χ²(15) ⟹  Pr > χ² = 0.0000
Formula: log(Earnings) ~ 1 + Educatn + Age + (Age ^ 2) + Kids
Variance Covariance Estimator: OIM
────────────────────────────────────────────────────────────────────────────────────────
                    PE           SE        t-value  Pr > |t|         2.50%        97.50%
────────────────────────────────────────────────────────────────────────────────────────
(Intercept)   8.55057      0.131389      65.0782      <1e-99   8.29306       8.80809
Educatn       0.000538588  0.000138739    3.88203     0.0001   0.000266663   0.000810512
Age           0.0389003    0.0068679      5.66408     <1e-07   0.0254394     0.0523613
Age ^ 2      -0.000191192  8.86411e-5    -2.15692     0.0310  -0.000364926  -1.74575e-5
Kids: 1      -0.132069     0.00828577   -15.9392      <1e-56  -0.148309     -0.115829
Kids: 2      -0.290339     0.00727643   -39.9013      <1e-99  -0.304601     -0.276078
Kids: 3      -0.529931     0.00820589   -64.5793      <1e-99  -0.546014     -0.513848
Kids: 4      -0.710859     0.011204     -63.447       <1e-99  -0.732818     -0.688899
Kids: 5      -0.491404     0.0180217    -27.2673      <1e-99  -0.526726     -0.456082
Kids: 6      -1.47718      0.0381049    -38.7663      <1e-99  -1.55187      -1.4025
Kids: 7      -0.242236     0.0609217     -3.97619     <1e-04  -0.361641     -0.122831
Kids: 8      -1.53682      0.328191      -4.68271     <1e-05  -2.18007      -0.893577
Kids: 9      -0.209419     1.13675       -0.184226    0.8538  -2.43741       2.01858
Kids: 10      0.62745      0.50838        1.23421     0.2171  -0.368963      1.62386
Kids: 98     -0.71333      0.0186111    -38.3281      <1e-99  -0.749807     -0.676853
Kids: 99     -1.02093      0.0252724    -40.3971      <1e-99  -1.07046      -0.971397
────────────────────────────────────────────────────────────────────────────────────────
```





### Absorbing features

If one only care about the estimates of a subset of features and controls such as number of kids (as categorical), one can absorb those features,

```julia
model = fit(EconometricModel,
            @formula(log(Earnings) ~ Educatn + Age + Age^2 + absorb(Kids)),
            data,
            wts = :PersNum)
```

```
Continuous Response Model
Number of observations: 223174
Null Loglikelihood: -6746726.41
Loglikelihood: -345253.18
R-squared: 0.0466
Wald: 860.46 ∼ F(3, 223158) ⟹ Pr > F = 0.0000
Formula: log(Earnings) ~ 1 + Educatn + Age + (Age ^ 2) + absorb(Kids)
Variance Covariance Estimator: OIM
──────────────────────────────────────────────────────────────────────────────────────
                    PE           SE      t-value  Pr > |t|         2.50%        97.50%
──────────────────────────────────────────────────────────────────────────────────────
(Intercept)   8.23837      0.13135      62.721      <1e-99   7.98093       8.49581
Educatn       0.000538588  0.000138739   3.88203    0.0001   0.000266663   0.000810512
Age           0.0389003    0.0068679     5.66408    <1e-07   0.0254394     0.0523613
Age ^ 2      -0.000191192  8.86411e-5   -2.15692    0.0310  -0.000364926  -1.74575e-5
──────────────────────────────────────────────────────────────────────────────────────
```





### Variance Covariance Estimators

Various variance-covariance estimators are available for continous response models.

For example,

```julia
vcov(model, HC0)
```

```
4×4 Hermitian{Float64, Matrix{Float64}}:
  0.000304618  -1.6099e-8    -1.60138e-5    2.07213e-7
 -1.6099e-8     2.51577e-10   6.23515e-10  -8.35769e-12
 -1.60138e-5    6.23515e-10   8.46239e-7   -1.10009e-8
  2.07213e-7   -8.35769e-12  -1.10009e-8    1.4367e-10
```



```julia
vcov(model, HC1)
```

```
4×4 Hermitian{Float64, Matrix{Float64}}:
  0.00030464  -1.61001e-8   -1.6015e-5    2.07228e-7
 -1.61001e-8   2.51595e-10   6.2356e-10  -8.35829e-12
 -1.6015e-5    6.2356e-10    8.463e-7    -1.10017e-8
  2.07228e-7  -8.35829e-12  -1.10017e-8   1.4368e-10
```



```julia
vcov(model, HC2)
```

```
4×4 Hermitian{Float64, Matrix{Float64}}:
  0.000305269  -1.62242e-8   -1.60486e-5    2.07673e-7
 -1.62242e-8    2.53522e-10   6.28483e-10  -8.41935e-12
 -1.60486e-5    6.28483e-10   8.48106e-7   -1.10257e-8
  2.07673e-7   -8.41935e-12  -1.10257e-8    1.43999e-10
```



```julia
vcov(model, HC3)
```

```
4×4 Hermitian{Float64, Matrix{Float64}}:
  0.000305927  -1.63508e-8   -1.60838e-5   2.08138e-7
 -1.63508e-8    2.55503e-10   6.3349e-10  -8.48145e-12
 -1.60838e-5    6.3349e-10    8.49996e-7  -1.10508e-8
  2.08138e-7   -8.48145e-12  -1.10508e-8   1.44333e-10
```



```julia
vcov(model, HC4)
```

```
4×4 Hermitian{Float64, Matrix{Float64}}:
  0.000307226  -1.66052e-8   -1.61533e-5    2.09058e-7
 -1.66052e-8    2.59567e-10   6.43482e-10  -8.60518e-12
 -1.61533e-5    6.43482e-10   8.53731e-7   -1.11003e-8
  2.09058e-7   -8.60518e-12  -1.11003e-8    1.44993e-10
```





## Longitudinal Data

Longitudinal data is when a sample contains repeated measurements of the unit of observation. This kind of data allows for various estimators that make use of such a structure.

Example,

```julia
data = RDatasets.dataset("Ecdat", "Crime") |>
  (data -> select(data, [:County, :Year, :CRMRTE, :PrbConv, :AvgSen, :PrbPris]))
```

```
630×6 DataFrame
 Row │ County  Year   CRMRTE     PrbConv   AvgSen   PrbPris
     │ Int32   Int32  Float64    Float64   Float64  Float64
─────┼───────────────────────────────────────────────────────
   1 │      1     81  0.0398849  0.402062     5.61  0.472222
   2 │      1     82  0.0383449  0.433005     5.59  0.506993
   3 │      1     83  0.0303048  0.525703     5.8   0.479705
   4 │      1     84  0.0347259  0.604706     6.89  0.520104
   5 │      1     85  0.036573   0.578723     6.55  0.497059
   6 │      1     86  0.0347524  0.512324     6.9   0.439863
   7 │      1     87  0.0356036  0.527596     6.71  0.43617
   8 │      3     81  0.0163921  0.869048     8.45  0.465753
  ⋮  │   ⋮       ⋮        ⋮         ⋮         ⋮        ⋮
 624 │    197     81  0.0178621  1.06098     11.35  0.356322
 625 │    197     82  0.0180711  0.545455     6.64  0.363636
 626 │    197     83  0.0155747  0.480392     7.77  0.428571
 627 │    197     84  0.0136619  1.41026     10.11  0.372727
 628 │    197     85  0.0130857  0.830769     5.96  0.333333
 629 │    197     86  0.012874   2.25         7.68  0.244444
 630 │    197     87  0.0141928  1.18293     12.23  0.360825
                                             615 rows omitted
```





### The Between estimator

```julia
model = fit(BetweenEstimator, # Indicates the between estimator
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data,
            panel = :County)
```

```
Between Estimator
County with 630 groups
Balanced groups with size 7
Number of observations: 90
Null Loglikelihood: 239.56
Loglikelihood: 244.40
R-squared: 0.1029
Wald: 3.29 ∼ F(3, 86) ⟹ Pr > F = 0.0246
Formula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris
Variance Covariance Estimator: OIM
───────────────────────────────────────────────────────────────────────────────────
                    PE          SE       t-value  Pr > |t|        2.50%      97.50%
───────────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.00464339   0.0186571   -0.24888     0.8040  -0.0417325   0.0324457
PrbConv      -0.00354113   0.0018904   -1.87322     0.0644  -0.00729911  0.00021685
AvgSen        0.000848049  0.00116477   0.728086    0.4685  -0.00146743  0.00316353
PrbPris       0.0730299    0.0332057    2.19932     0.0305   0.0070191   0.139041
───────────────────────────────────────────────────────────────────────────────────
```





### The Within estimator (Fixed Effects)

```julia
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County)),
            data)
```

```
Continuous Response Model
Number of observations: 630
Null Loglikelihood: 1633.30
Loglikelihood: 2273.95
R-squared: 0.8707
Wald: 0.16 ∼ F(3, 537) ⟹ Pr > F = 0.9258
Formula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris + absorb(County)
Variance Covariance Estimator: OIM
──────────────────────────────────────────────────────────────────────────────────────
                   PE           SE        t-value  Pr > |t|         2.50%       97.50%
──────────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.0314527   0.00204638   15.3699       <1e-43   0.0274328    0.0354726
PrbConv       6.65981e-6  0.0001985     0.0335507    0.9732  -0.000383272  0.000396592
AvgSen        7.83181e-5  0.000127904   0.612318     0.5406  -0.000172936  0.000329572
PrbPris      -0.0013419   0.00405182   -0.331185     0.7406  -0.00930126   0.00661746
──────────────────────────────────────────────────────────────────────────────────────
```





Absorbing the panel and temporal indicators

```julia
model = fit(EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County + Year)),
            data)
```

```
Continuous Response Model
Number of observations: 630
Null Loglikelihood: 1633.30
Loglikelihood: 2290.25
R-squared: 0.8775
Wald: 0.23 ∼ F(3, 531) ⟹ Pr > F = 0.8767
Formula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris + absorb(County + Year)
Variance Covariance Estimator: OIM
──────────────────────────────────────────────────────────────────────────────────────
                    PE           SE       t-value  Pr > |t|         2.50%       97.50%
──────────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.0319651    0.00206378   15.4886      <1e-44   0.027911     0.0360193
PrbConv       7.90362e-5   0.000196089   0.403062    0.6871  -0.00030617   0.000464242
AvgSen       -9.4788e-5    0.000134924  -0.702531    0.4827  -0.000359838  0.000170262
PrbPris       0.000979533  0.0040432     0.242267    0.8087  -0.00696309   0.00892216
──────────────────────────────────────────────────────────────────────────────────────
```





### The random effects model à la Swamy-Arora

```julia
model = fit(RandomEffectsEstimator, # Indicates the random effects estimator
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data,
            panel = :County,
            time = :Year)
```

```
One-way Random Effect Model
Longitudinal dataset: County, Year
Balanced dataset with 90 panels of length 7
individual error component: 0.0162
idiosyncratic error component: 0.0071
ρ: 0.8399
Number of observations: 630
Null Loglikelihood: 2225.79
Loglikelihood: 2226.01
R-squared: 0.0007
Wald: 0.15 ∼ F(3, 626) ⟹ Pr > F = 0.9291
Formula: CRMRTE ~ PrbConv + AvgSen + PrbPris
Variance Covariance Estimator: OIM
──────────────────────────────────────────────────────────────────────────────────────
                    PE           SE       t-value  Pr > |t|         2.50%       97.50%
──────────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.0309293    0.00266706   11.5968      <1e-27   0.0256918    0.0361668
PrbConv      -3.96634e-5   0.000198471  -0.199845    0.8417  -0.000429413  0.000350086
AvgSen        8.2428e-5    0.000127818   0.644887    0.5192  -0.000168575  0.000333431
PrbPris      -0.000123327  0.00404272   -0.030506    0.9757  -0.00806226   0.00781561
──────────────────────────────────────────────────────────────────────────────────────
```





## Instrumental Variables

Instrumental variables models are estimated using the 2SLS estimator.

```julia
model = fit(RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
            data,
            panel = :County,
            time = :Year)
```

```
One-way Random Effect Model
Longitudinal dataset: County, Year
Balanced dataset with 90 panels of length 7
individual error component: 0.0413
idiosyncratic error component: 0.0074
ρ: 0.9691
Number of observations: 630
Null Loglikelihood: 2268.01
Loglikelihood: 2248.34
R-squared: NaN
Wald: 0.03 ∼ F(2, 626) ⟹ Pr > F = 0.9671
Formula: CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)
Variance Covariance Estimator: OIM
───────────────────────────────────────────────────────────────────────────────────────
                    PE           SE        t-value  Pr > |t|         2.50%       97.50%
───────────────────────────────────────────────────────────────────────────────────────
(Intercept)   0.037747     0.0241684     1.56183      0.1188  -0.00971405   0.085208
PrbConv       1.39581e-5   0.000200244   0.0697054    0.9445  -0.000379273  0.000407189
AvgSen       -0.000688924  0.00266514   -0.258494     0.7961  -0.00592262   0.00454478
───────────────────────────────────────────────────────────────────────────────────────
```





## Nominal Response Models

```julia
data = CSV.read(joinpath(pkgdir(Econometrics), "data", "insure.csv"), DataFrame) |>
   (data -> select(data, [:insure, :age, :male, :nonwhite, :site])) |>
   dropmissing |>
   (data -> transform!(data, [:insure, :site] .=> categorical, renamecols = false))
```

```
615×5 DataFrame
 Row │ insure     age      male   nonwhite  site
     │ Cat…       Float64  Int64  Int64     Cat…
─────┼───────────────────────────────────────────
   1 │ Indemnity  73.7221      0         0  2
   2 │ Prepaid    27.8959      0         0  2
   3 │ Indemnity  37.5414      0         0  1
   4 │ Prepaid    23.6413      0         1  3
   5 │ Prepaid    29.6838      0         0  2
   6 │ Prepaid    39.4689      0         0  2
   7 │ Uninsure   63.102       0         1  3
   8 │ Prepaid    69.8398      0         0  1
  ⋮  │     ⋮         ⋮       ⋮       ⋮       ⋮
 609 │ Prepaid    69.9877      0         0  1
 610 │ Prepaid    71.5209      0         0  3
 611 │ Prepaid    63.4168      0         0  2
 612 │ Prepaid    40.3532      0         1  1
 613 │ Prepaid    55.3237      1         0  3
 614 │ Prepaid    27.9836      1         0  3
 615 │ Prepaid    30.59        1         0  3
                                 600 rows omitted
```





The model automatically detects that the response is nominal and uses the correct model.

```julia
model = fit(EconometricModel,
            @formula(insure ~ age + male + nonwhite + site),
            data,
            contrasts = Dict(:insure => DummyCoding(base = "Uninsure")))
```

```
Probability Model for Nominal Response
Categories: Uninsure, Indemnity, Prepaid
Number of observations: 615
Null Loglikelihood: -555.85
Loglikelihood: -534.36
R-squared: 0.0387
LR Test: 42.99 ∼ χ²(10) ⟹ Pr > χ² = 0.0000
Formula: insure ~ 1 + age + male + nonwhite + site
───────────────────────────────────────────────────────────────────────────────────────────────────
                                       PE         SE       t-value  Pr > |t|       2.50%     97.50%
───────────────────────────────────────────────────────────────────────────────────────────────────
insure: Indemnity ~ (Intercept)   1.28694     0.59232     2.17271     0.0302   0.123689   2.4502
insure: Indemnity ~ age           0.00779612  0.0114418   0.681372    0.4959  -0.0146743  0.0302666
insure: Indemnity ~ male         -0.451848    0.367486   -1.22957     0.2193  -1.17355    0.269855
insure: Indemnity ~ nonwhite     -0.217059    0.425636   -0.509965    0.6103  -1.05296    0.618843
insure: Indemnity ~ site: 2       1.21152     0.470506    2.57493     0.0103   0.287497   2.13554
insure: Indemnity ~ site: 3       0.207813    0.366293    0.56734     0.5707  -0.511547   0.927172
insure: Prepaid ~ (Intercept)     1.55666     0.596327    2.61041     0.0093   0.385533   2.72778
insure: Prepaid ~ age            -0.00394887  0.0115993  -0.340439    0.7336  -0.0267287  0.018831
insure: Prepaid ~ male            0.109846    0.365187    0.300793    0.7637  -0.607343   0.827035
insure: Prepaid ~ nonwhite        0.757718    0.419575    1.80592     0.0714  -0.0662835  1.58172
insure: Prepaid ~ site: 2         1.32456     0.469789    2.81947     0.0050   0.401941   2.24717
insure: Prepaid ~ site: 3        -0.380175    0.372819   -1.01973     0.3083  -1.11235    0.352001
───────────────────────────────────────────────────────────────────────────────────────────────────
```





## Ordinal Response Models

```julia
data = RDatasets.dataset("Ecdat", "Kakadu") |>
       (data -> select(data, [:RecParks, :Sex, :Age, :Schooling]))
data.RecParks = convert(Vector{Int}, data.RecParks)
data.RecParks = levels!(categorical(data.RecParks, ordered = true, compress = true), collect(1:5))
data
```

```
1827×4 DataFrame
  Row │ RecParks  Sex     Age    Schooling
      │ Cat…      Cat…    Int32  Int32
──────┼────────────────────────────────────
    1 │ 3         male       27          3
    2 │ 5         female     32          4
    3 │ 4         male       32          4
    4 │ 1         female     70          6
    5 │ 2         male       32          5
    6 │ 3         male       47          6
    7 │ 1         male       42          5
    8 │ 5         female     70          3
  ⋮   │    ⋮        ⋮       ⋮        ⋮
 1821 │ 4         male       21          5
 1822 │ 3         male       21          3
 1823 │ 3         male       52          4
 1824 │ 2         female     21          3
 1825 │ 5         female     21          2
 1826 │ 1         female     21          4
 1827 │ 4         male       21          5
                          1812 rows omitted
```





The model automatically detects that the response is ordinal and uses the correct model.

```julia
model = fit(EconometricModel,
            @formula(RecParks ~ Age + Sex + Schooling),
            data)
```

```
Probability Model for Ordinal Response
Categories: 1 < 2 < 3 < 4 < 5
Number of observations: 1827
Null Loglikelihood: -2677.60
Loglikelihood: -2657.40
R-squared: 0.0075
LR Test: 40.42 ∼ χ²(3) ⟹ Pr > χ² = 0.0000
Formula: RecParks ~ Age + Sex + Schooling
──────────────────────────────────────────────────────────────────────────────────────────
                          PE          SE        t-value  Pr > |t|        2.50%      97.50%
──────────────────────────────────────────────────────────────────────────────────────────
Age                  0.00943647  0.00254031    3.71469     0.0002   0.00445424   0.0144187
Sex: male           -0.0151659   0.0846365    -0.179188    0.8578  -0.181161     0.150829
Schooling           -0.103902    0.0248742    -4.17711     <1e-04  -0.152687    -0.0551174
(Intercept): 1 | 2  -2.92405     0.191927    -15.2352      <1e-48  -3.30047     -2.54763
(Intercept): 2 | 3  -1.54922     0.171444     -9.03632     <1e-18  -1.88547     -1.21297
(Intercept): 3 | 4  -0.298938    0.166904     -1.79108     0.0734  -0.626281     0.0284051
(Intercept): 4 | 5   0.669835    0.167622      3.9961      <1e-04   0.341083     0.998587
──────────────────────────────────────────────────────────────────────────────────────────
```


