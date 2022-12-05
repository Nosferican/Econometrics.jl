# Linear Models
@testset "Linear Models" begin
    @testset "Balanced Panel Data" begin
        # Balanced Panel Data
        data = dataset("Ecdat", "Crime")
        # Pooling
        model = fit(EconometricModel, @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris), data)
        @test sprint(show, model) ==
            "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 1643.30\nR-squared: 0.0312\nLR Test: 19.99 ∼ χ²(3) ⟹  Pr > χ² = 0.0002\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE      t-value  Pr > |t|         2.50%        97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0186413    0.00431066    4.32447    <1e-04   0.0101762     0.0271064\nPrbConv      -0.00116473   0.000422062  -2.75961    0.0060  -0.00199356   -0.0003359\nAvgSen        0.000236185  0.000268216   0.88058    0.3789  -0.000290526   0.000762897\nPrbPris       0.0273394    0.00817642    3.34369    0.0009   0.0112829     0.0433959\n──────────────────────────────────────────────────────────────────────────────────────"
        @test nullloglikelihood(model) ≈ 1633.305 rtol = 1e-3
        @test loglikelihood(model) ≈ 1643.304 rtol = 1e-3
        @test dof(model) == 5
        # This corresponds to BIC from R's lm, Stata uses rank(vcov(model))
        @test bic(model) ≈ -3254.379 rtol = 1e-3
        @test coef(model) ≈ [0.0186413, -0.0011647, 0.0002362, 0.0273394] rtol = 1e-5
        @test vcov(model) ≈ [
            1.858177e-05 -1.620480e-07 -6.464872e-07 -2.860941e-05
            -1.620480e-07 1.781365e-07 -1.715589e-09 1.286486e-07
            -6.464872e-07 -1.715589e-09 7.193971e-08 8.182148e-09
            -2.860941e-05 1.286486e-07 8.182148e-09 6.685387e-05
            ] rtol = 1e-6
        @test confint(model) ≈ [
            0.0101761950 0.0271063941
            -0.0019935581 -0.0003358998
            -0.0002905263 0.0007628970
            0.0112828681 0.0433959412
            ] rtol = 1e-6
        # Between
        model = fit(
            BetweenEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data,
            panel = :County,
            )
        @test sprint(show, model) ==
            "Between Estimator\nCounty with 630 groups\nBalanced groups with size 7\nNumber of observations: 90\nNull Loglikelihood: 239.56\nLoglikelihood: 244.40\nR-squared: 0.1029\nWald: 3.29 ∼ F(3, 86) ⟹ Pr > F = 0.0246\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris\nVariance Covariance Estimator: OIM\n───────────────────────────────────────────────────────────────────────────────────\n                    PE          SE       t-value  Pr > |t|        2.50%      97.50%\n───────────────────────────────────────────────────────────────────────────────────\n(Intercept)  -0.00464339   0.0186571   -0.24888     0.8040  -0.0417325   0.0324457\nPrbConv      -0.00354113   0.0018904   -1.87322     0.0644  -0.00729911  0.00021685\nAvgSen        0.000848049  0.00116477   0.728086    0.4685  -0.00146743  0.00316353\nPrbPris       0.0730299    0.0332057    2.19932     0.0305   0.0070191   0.139041\n───────────────────────────────────────────────────────────────────────────────────"
        @test nullloglikelihood(model) ≈ 239.5638 rtol = 1e-3
        @test loglikelihood(model) ≈ 244.4479 rtol = 1e-3
        @test dof(model) == 5
        @test coef(model) ≈ [-0.00464339, -0.00354113, 0.00084805, 0.07302994] rtol = 1e-6
        @test vcov(model) ≈ [
            3.480883e-04 -5.985740e-06 -1.382250e-05 -5.104706e-04
            -5.985740e-06 3.573593e-06 -1.212431e-08 8.538922e-06
            -1.382250e-05 -1.212431e-08 1.356678e-06 3.953835e-06
            -5.104706e-04 8.538922e-06 3.953835e-06 1.102622e-03
            ] rtol = 1e-6
        # Using Stata's value, R's plm has confint incorrect
        @test confint(model) ≈ [
            -0.0417325 0.0324457
            -0.0072991 0.0002169
            -0.0014674 0.0031635
            0.0070191 0.1390408
            ] rtol = 1e-5
        # Within
        model = fit(
            EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County)),
            data,
            )
        @test sprint(show, model) ==
            "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2273.95\nR-squared: 0.8707\nWald: 0.16 ∼ F(3, 537) ⟹ Pr > F = 0.9258\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris + absorb(County)\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                   PE           SE        t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0314527   0.00204638   15.3699       <1e-43   0.0274328    0.0354726\nPrbConv       6.65981e-6  0.0001985     0.0335507    0.9732  -0.000383272  0.000396592\nAvgSen        7.83181e-5  0.000127904   0.612318     0.5406  -0.000172936  0.000329572\nPrbPris      -0.0013419   0.00405182   -0.331185     0.7406  -0.00930126   0.00661746\n──────────────────────────────────────────────────────────────────────────────────────"
        model = fit(
            EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County + Year)),
            data,
            )
        @test sprint(show, model) ==
            "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2290.25\nR-squared: 0.8775\nWald: 0.23 ∼ F(3, 531) ⟹ Pr > F = 0.8767\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris + absorb(County + Year)\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE       t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0319651    0.00206378   15.4886      <1e-44   0.027911     0.0360193\nPrbConv       7.90362e-5   0.000196089   0.403062    0.6871  -0.00030617   0.000464242\nAvgSen       -9.4788e-5    0.000134924  -0.702531    0.4827  -0.000359838  0.000170262\nPrbPris       0.000979533  0.0040432     0.242267    0.8087  -0.00696309   0.00892216\n──────────────────────────────────────────────────────────────────────────────────────"
        # Random Effects
        model = fit(
            RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
            data,
            panel = :County,
            time = :Year,
            )
        @test sprint(show, model) ==
            "One-way Random Effect Model\nLongitudinal dataset: County, Year\nBalanced dataset with 90 panels of length 7\nindividual error component: 0.0162\nidiosyncratic error component: 0.0071\nρ: 0.8399\nNumber of observations: 630\nNull Loglikelihood: 2225.79\nLoglikelihood: 2226.01\nR-squared: 0.0007\nWald: 0.15 ∼ F(3, 626) ⟹ Pr > F = 0.9291\nFormula: CRMRTE ~ PrbConv + AvgSen + PrbPris\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE       t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0309293    0.00266706   11.5968      <1e-27   0.0256918    0.0361668\nPrbConv      -3.96634e-5   0.000198471  -0.199845    0.8417  -0.000429413  0.000350086\nAvgSen        8.2428e-5    0.000127818   0.644887    0.5192  -0.000168575  0.000333431\nPrbPris      -0.000123327  0.00404272   -0.030506    0.9757  -0.00806226   0.00781561\n──────────────────────────────────────────────────────────────────────────────────────"
        wagepan = DataFrame(wooldridge("wagepan"))
        # From #83
        model = fit(
            RandomEffectsEstimator,
            @formula(lwage ~ educ + black + hisp + exper + exper^2 + married + union + year),
            wagepan,
            panel = :nr,
            time = :year,
            contrasts = Dict(:year => DummyCoding())
            )
        @test coef(model) ≈ [
            0.023586377, 0.091876276, -0.139376726, 0.021731732, 0.105754520,
            -0.004723943, 0.063986022, 0.106134429, 0.040462003, 0.030921157,
            0.020280640, 0.043118708, 0.057815458, 0.091947584, 0.134928917
            ] rtol = 1e-3
        # IV Pooling
        model = fit(EconometricModel, @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)), data)
        @test sprint(show, model) ==
            "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: -611.40\nR-squared: NaN\nLR Test: -4489.40 ∼ χ²(2) ⟹  Pr > χ² = 1.0000\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris)\nVariance Covariance Estimator: OIM\n─────────────────────────────────────────────────────────────────────────────────\n                   PE          SE        t-value  Pr > |t|       2.50%     97.50%\n─────────────────────────────────────────────────────────────────────────────────\n(Intercept)   2.17878     23.0244      0.0946291    0.9246  -43.0357    47.3932\nPrbConv       0.00456765   0.0638118   0.0715801    0.9430   -0.120743   0.129879\nAvgSen       -0.240139     2.57602    -0.093221     0.9258   -5.29883    4.81855\n─────────────────────────────────────────────────────────────────────────────────"
        # Full ranking instruments #72
        data = CSV.read(
            joinpath(pkgdir(Econometrics), "data", "_72.csv"),
            DataFrame
            )
        model = fit(
            EconometricModel,
            @formula(V ~ 1 + (d + p ~ (c1 + c2 + r2 + p̃) * j)),
            data)
        @test isa(model, EconometricModel)
        # IV Between
        model = fit(
            BetweenEstimator,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
            data,
            panel = :County,
            )
        @test sprint(show, model) ==
            "Between Estimator\nCounty with 630 groups\nBalanced groups with size 7\nNumber of observations: 90\nNull Loglikelihood: 239.56\nLoglikelihood: 161.00\nR-squared: NaN\nWald: 0.73 ∼ F(2, 86) ⟹ Pr > F = 0.4844\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris)\nVariance Covariance Estimator: OIM\n─────────────────────────────────────────────────────────────────────────────────\n                   PE          SE       t-value  Pr > |t|       2.50%      97.50%\n─────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.250667    0.255826     0.979834    0.3299  -0.257898   0.759233\nPrbConv      -0.00331719  0.00481737  -0.688589    0.4929  -0.0128938  0.00625942\nAvgSen       -0.0242107   0.0286331   -0.845549    0.4002  -0.0811314  0.03271\n─────────────────────────────────────────────────────────────────────────────────"
        # IV Within
        model = fit(
            EconometricModel,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County)),
            data,
            )
        @test sprint(show, model) ==
            "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2245.89\nR-squared: NaN\nWald: 0.04 ∼ F(2, 537) ⟹ Pr > F = 0.9583\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris) + absorb(County)\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE       t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0389703    0.025508      1.52777     0.1272  -0.0111374    0.089078\nPrbConv       2.46646e-5   0.000215805   0.114291    0.9090  -0.000399261  0.000448591\nAvgSen       -0.000826364  0.00285293   -0.289654    0.7722  -0.00643064   0.00477791\n──────────────────────────────────────────────────────────────────────────────────────"
        model = fit(
            EconometricModel,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County + Year)),
            data,
            )
        @test sprint(show, model) ==
            "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2280.31\nR-squared: NaN\nWald: 0.09 ∼ F(2, 531) ⟹ Pr > F = 0.9116\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris) + absorb(County + Year)\nVariance Covariance Estimator: OIM\n────────────────────────────────────────────────────────────────────────────────────\n                   PE           SE      t-value  Pr > |t|         2.50%       97.50%\n────────────────────────────────────────────────────────────────────────────────────\n(Intercept)  0.027409     0.0208187    1.31656     0.1886  -0.0134881    0.0683061\nPrbConv      6.00404e-5   0.000214966  0.279301    0.7801  -0.000362248  0.000482329\nAvgSen       0.000462028  0.0023309    0.198219    0.8429  -0.00411688   0.00504094\n────────────────────────────────────────────────────────────────────────────────────"
        # IV Random
        model = fit(
            RandomEffectsEstimator,
            @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
            data,
            panel = :County,
            time = :Year,
            )
        @test sprint(show, model) ==
            "One-way Random Effect Model\nLongitudinal dataset: County, Year\nBalanced dataset with 90 panels of length 7\nindividual error component: 0.0413\nidiosyncratic error component: 0.0074\nρ: 0.9691\nNumber of observations: 630\nNull Loglikelihood: 2268.01\nLoglikelihood: 2248.34\nR-squared: NaN\nWald: 0.03 ∼ F(2, 626) ⟹ Pr > F = 0.9671\nFormula: CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)\nVariance Covariance Estimator: OIM\n───────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE        t-value  Pr > |t|         2.50%       97.50%\n───────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.037747     0.0241684     1.56183      0.1188  -0.00971405   0.085208\nPrbConv       1.39581e-5   0.000200244   0.0697054    0.9445  -0.000379273  0.000407189\nAvgSen       -0.000688924  0.00266514   -0.258494     0.7961  -0.00592262   0.00454478\n───────────────────────────────────────────────────────────────────────────────────────"
    end
    @testset "Heteroscedasticity Consistent Variance Covariance Estimators" begin
        data = dataset("Ecdat", "Crime")
        # Pooling
        model = fit(EconometricModel, @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris), data)
        @test vcov(model) ≈ vcov(model, OIM)
        @test vcov(model, HC0) ≈ [
            1.51708E-05 3.58518E-07 -7.30083E-07 -2.03240E-05
            3.58518E-07 2.85571E-07 -3.58783E-08 -4.13434E-07
            -7.30083E-07 -3.58783E-08 9.36232E-08 -1.46277E-07
            -2.03240E-05 -4.13434E-07 -1.46277E-07 5.17390E-05
            ] rtol = 1e-6
        @test vcov(model, HC1) ≈ [
            1.52677E-05 3.60808E-07 -7.34748E-07 -2.04539E-05
            3.60808E-07 2.87396E-07 -3.61076E-08 -4.16076E-07
            -7.34748E-07 -3.61076E-08 9.42214E-08 -1.47212E-07
            -2.04539E-05 -4.16076E-07 -1.47212E-07 5.20696E-05
            ] rtol = 1e-6
        @test vcov(model, HC2) ≈ [
            1.56208E-05 9.05631E-08 -7.36000E-07 -2.08761E-05
            9.05631E-08 6.78782E-07 -5.03354E-08 -3.93296E-08
            -7.36000E-07 -5.03354E-08 9.59715E-08 -1.59383E-07
            -2.08761E-05 -3.93296E-08 -1.59383E-07 5.27586E-05
            ] rtol = 1e-6
        @test vcov(model, HC3) ≈ [
            1.66839E-05 -9.88963E-07 -7.13420E-07 -2.23048E-05
            -9.88963E-07 2.16675E-06 -1.04165E-07 1.50077E-06
            -7.13420E-07 -1.04165E-07 9.98145E-08 -2.14382E-07
            -2.23048E-05 1.50077E-06 -2.14382E-07 5.50463E-05
            ] rtol = 1e-6
        @test vcov(model, HC4) ≈ [
            3.21597E-05 -2.14119E-05 -4.18897E-09 -4.43245E-05
            -2.14119E-05 2.97197E-05 -1.09486E-06 3.08355E-05
            -4.18897E-09 -1.09486E-06 1.38192E-07 -1.26498E-06
            -4.43245E-05 3.08355E-05 -1.26498E-06 8.67924E-05
            ] rtol = 1e-6
        dv = fit(
            EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + County),
            data,
            contrasts = Dict(:County => DummyCoding()),
            )
        fe = fit(
            EconometricModel,
            @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County)),
            data,
            )
        @test vcov(fe)[2:end, 2:end] ≈ vcov(dv)[2:4, 2:4]
        @test vcov(fe, HC1)[2:end, 2:end] ≈ vcov(dv, HC1)[2:4, 2:4]
    end
end
