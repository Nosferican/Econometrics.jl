using Econometrics, Test
using Econometrics: ConvergenceException, Hermitian
using CSV, RDatasets
# Exceptions
@testset "Exceptions" begin
	data = dataset("Ecdat", "Crime")
	data.AvgSen2 = 2 * data.AvgSen
	# Rank-deficient model
	@test isa(fit(EconometricModel,
			      @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + AvgSen2),
				  data),
			  EconometricModel)
	@test isa(fit(RandomEffectsEstimator,
				  @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
				  data,
				  panel = :County,
				  time = :Year),
			  EconometricModel)
	# Model specifics
	@test_throws(ArgumentError("The between estimator requires the panel identifier."),
				 fit(BetweenEstimator,
				 	 @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
					 data))
	@test_throws(ArgumentError("Absorbing features is only implemented for least squares."),
				 fit(BetweenEstimator,
				 	 @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
					 data,
					 panel = :County))
	@test_throws(ArgumentError("The random effects estimator requires the panel and temporal identifiers."),
				 fit(RandomEffectsEstimator,
				 	 @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
					 data,
					 panel = :County))
	@test_throws(ArgumentError("Absorbing features is only implemented for least squares."),
				 fit(RandomEffectsEstimator,
				 	 @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(Year)),
					 data,
					 panel = :County,
					 time = :Year))
	data = dataset("datasets", "iris")
	@test_throws(ConvergenceException,
				 fit(EconometricModel,
				 	 @formula(Species ~ SepalLength + SepalWidth),
				 	 data))
	# Robust variance covariance estimators
	@test_throws(ArgumentError("Robust variance covariance estimators only available for continous response models."),
				 fit(EconometricModel,
				 	 @formula(Species ~ SepalLength + SepalWidth),
				 	 data,
					 vce = HC1))
	@test_throws(ArgumentError("When absorbing features, only HC2-HC4 are unavailable."),
				 fit(EconometricModel,
				 	 @formula(SepalLength ~ SepalLength + absorb(Species)),
				 	 data,
					 vce = HC2))
end
# Linear Models
@testset "Linear Models" begin
	@testset "Balanced Panel Data" begin
    	# Balanced Panel Data
    	data = dataset("Ecdat", "Crime")
    	# Pooling
    	model = fit(EconometricModel,
                	@formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
                	data)
		@test sprint(show, model) == "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 1643.30\nR-squared: 0.0312\nLR Test: 19.99 ∼ χ²(3) ⟹  Pr > χ² = 0.0002\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE      t-value  Pr > |t|         2.50%        97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0186413    0.00431066    4.32447    <1e-4    0.0101762     0.0271064  \nPrbConv      -0.00116473   0.000422062  -2.75961    0.0060  -0.00199356   -0.0003359  \nAvgSen        0.000236185  0.000268216   0.88058    0.3789  -0.000290526   0.000762897\nPrbPris       0.0273394    0.00817642    3.34369    0.0009   0.0112829     0.0433959  \n──────────────────────────────────────────────────────────────────────────────────────"
    	@test nullloglikelihood(model) ≈ 1633.305 rtol = 1e-3
    	@test loglikelihood(model) ≈ 1643.304 rtol = 1e-3
    	@test dof(model) == 5
    	# This corresponds to BIC from R's lm, Stata uses rank(vcov(model))
    	@test bic(model) ≈ -3254.379 rtol = 1e-3
    	@test coef(model) ≈ [0.0186413, -0.0011647, 0.0002362, 0.0273394] rtol = 1e-5
    	@test vcov(model) ≈ [ 1.858177e-05 -1.620480e-07 -6.464872e-07 -2.860941e-05
                          	 -1.620480e-07  1.781365e-07 -1.715589e-09  1.286486e-07
                          	 -6.464872e-07 -1.715589e-09  7.193971e-08  8.182148e-09
                             -2.860941e-05  1.286486e-07  8.182148e-09  6.685387e-05
                        	] rtol = 1e-6
    	@test confint(model) ≈ [ 0.0101761950  0.0271063941
                            	-0.0019935581 -0.0003358998
                            	-0.0002905263  0.0007628970
                             	0.0112828681  0.0433959412
                           	   ] rtol = 1e-6
    	# Between
    	model = fit(BetweenEstimator,
                	@formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
                	data,
					panel = :County)
		@test sprint(show, model) == "Between Estimator\nCounty with 630 groups\nBalanced groups with size 7\nNumber of observations: 90\nNull Loglikelihood: 239.56\nLoglikelihood: 244.40\nR-squared: 0.1029\nWald: 3.29 ∼ F(3, 86) ⟹ Pr > F = 0.0246\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris\nVariance Covariance Estimator: OIM\n───────────────────────────────────────────────────────────────────────────────────\n                    PE          SE       t-value  Pr > |t|        2.50%      97.50%\n───────────────────────────────────────────────────────────────────────────────────\n(Intercept)  -0.00464339   0.0186571   -0.24888     0.8040  -0.0417325   0.0324457 \nPrbConv      -0.00354113   0.0018904   -1.87322     0.0644  -0.00729911  0.00021685\nAvgSen        0.000848049  0.00116477   0.728086    0.4685  -0.00146743  0.00316353\nPrbPris       0.0730299    0.0332057    2.19932     0.0305   0.0070191   0.139041  \n───────────────────────────────────────────────────────────────────────────────────"
    	@test nullloglikelihood(model) ≈ 239.5638 rtol = 1e-3
    	@test loglikelihood(model) ≈ 244.4479 rtol = 1e-3
    	@test dof(model) == 5
    	@test coef(model) ≈ [-0.00464339, -0.00354113, 0.00084805, 0.07302994] rtol = 1e-6
    	@test vcov(model) ≈ [ 3.480883e-04 -5.985740e-06 -1.382250e-05 -5.104706e-04
                         	 -5.985740e-06  3.573593e-06 -1.212431e-08  8.538922e-06
                         	 -1.382250e-05 -1.212431e-08  1.356678e-06  3.953835e-06
                         	 -5.104706e-04  8.538922e-06  3.953835e-06  1.102622e-03
                        	] rtol = 1e-6
    	# Using Stata's value, R's plm has confint incorrect
    	@test confint(model) ≈ [-0.0417325 0.0324457
                            	-0.0072991 0.0002169
                            	-0.0014674 0.0031635
                             	 0.0070191 0.1390408
                           	   ] rtol = 1e-5
    	# Within
    	model = fit(EconometricModel,
                	@formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County)),
                	data)
		@test sprint(show, model) == "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2273.95\nR-squared: 0.8707\nWald: 0.16 ∼ F(3, 537) ⟹ Pr > F = 0.9258\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris + absorb(County)\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                   PE           SE        t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0314527   0.00204638   15.3699       <1e-43   0.0274328    0.0354726  \nPrbConv       6.65981e-6  0.0001985     0.0335507    0.9732  -0.000383272  0.000396592\nAvgSen        7.83181e-5  0.000127904   0.612318     0.5406  -0.000172936  0.000329572\nPrbPris      -0.0013419   0.00405182   -0.331185     0.7406  -0.00930126   0.00661746 \n──────────────────────────────────────────────────────────────────────────────────────"
    	model = fit(EconometricModel,
                	@formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County + Year)),
                	data)
		@test sprint(show, model) == "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2290.25\nR-squared: 0.8775\nWald: 0.23 ∼ F(3, 531) ⟹ Pr > F = 0.8767\nFormula: CRMRTE ~ 1 + PrbConv + AvgSen + PrbPris + absorb(County + Year)\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE       t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0319651    0.00206378   15.4886      <1e-44   0.027911     0.0360193  \nPrbConv       7.90362e-5   0.000196089   0.403062    0.6871  -0.00030617   0.000464242\nAvgSen       -9.4788e-5    0.000134924  -0.702531    0.4827  -0.000359838  0.000170262\nPrbPris       0.000979533  0.0040432     0.242267    0.8087  -0.00696309   0.00892216 \n──────────────────────────────────────────────────────────────────────────────────────"
    	# Random Effects
    	model = fit(RandomEffectsEstimator,
                	@formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
                	data,
					panel = :County,
					time = :Year)
		@test sprint(show, model) == "One-way Random Effect Model\nLongitudinal dataset: County, Year\nBalanced dataset with 90 panels of length 7\nindividual error component: 0.0162\nidiosyncratic error component: 0.0071\nρ: 0.8399\nNumber of observations: 630\nNull Loglikelihood: 2225.79\nLoglikelihood: 2226.01\nR-squared: 0.0007\nWald: 0.15 ∼ F(3, 626) ⟹ Pr > F = 0.9291\nFormula: CRMRTE ~ PrbConv + AvgSen + PrbPris\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE       t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0309293    0.00266706   11.5968      <1e-27   0.0256918    0.0361668  \nPrbConv      -3.96634e-5   0.000198471  -0.199845    0.8417  -0.000429413  0.000350086\nAvgSen        8.2428e-5    0.000127818   0.644887    0.5192  -0.000168575  0.000333431\nPrbPris      -0.000123327  0.00404272   -0.030506    0.9757  -0.00806226   0.00781561 \n──────────────────────────────────────────────────────────────────────────────────────"
		# IV Pooling
    	model = fit(EconometricModel,
                	@formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
                	data)
		@test sprint(show, model) == "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: -611.40\nR-squared: NaN\nLR Test: -4489.40 ∼ χ²(2) ⟹  Pr > χ² = 1.0000\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris)\nVariance Covariance Estimator: OIM\n─────────────────────────────────────────────────────────────────────────────────\n                   PE          SE        t-value  Pr > |t|       2.50%     97.50%\n─────────────────────────────────────────────────────────────────────────────────\n(Intercept)   2.17878     23.0244      0.0946291    0.9246  -43.0357    47.3932  \nPrbConv       0.00456765   0.0638118   0.0715801    0.9430   -0.120743   0.129879\nAvgSen       -0.240139     2.57602    -0.093221     0.9258   -5.29883    4.81855 \n─────────────────────────────────────────────────────────────────────────────────"
    	# IV Between
    	model = fit(BetweenEstimator,
                	@formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
                	data,
					panel = :County)
		@test sprint(show, model) == "Between Estimator\nCounty with 630 groups\nBalanced groups with size 7\nNumber of observations: 90\nNull Loglikelihood: 239.56\nLoglikelihood: 161.00\nR-squared: NaN\nWald: 0.73 ∼ F(2, 86) ⟹ Pr > F = 0.4844\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris)\nVariance Covariance Estimator: OIM\n─────────────────────────────────────────────────────────────────────────────────\n                   PE          SE       t-value  Pr > |t|       2.50%      97.50%\n─────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.250667    0.255826     0.979834    0.3299  -0.257898   0.759233  \nPrbConv      -0.00331719  0.00481737  -0.688589    0.4929  -0.0128938  0.00625942\nAvgSen       -0.0242107   0.0286331   -0.845549    0.4002  -0.0811314  0.03271   \n─────────────────────────────────────────────────────────────────────────────────"
    	# IV Within
    	model = fit(EconometricModel,
	 				@formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County)),
                	data)
		@test sprint(show, model) == "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2245.89\nR-squared: NaN\nWald: 0.04 ∼ F(2, 537) ⟹ Pr > F = 0.9583\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris) + absorb(County)\nVariance Covariance Estimator: OIM\n──────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE       t-value  Pr > |t|         2.50%       97.50%\n──────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.0389703    0.025508      1.52777     0.1272  -0.0111374    0.089078   \nPrbConv       2.46646e-5   0.000215805   0.114291    0.9090  -0.000399261  0.000448591\nAvgSen       -0.000826364  0.00285293   -0.289654    0.7722  -0.00643064   0.00477791 \n──────────────────────────────────────────────────────────────────────────────────────"
    	model = fit(EconometricModel,
                	@formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County + Year)),
					data)
		@test sprint(show, model) == "Continuous Response Model\nNumber of observations: 630\nNull Loglikelihood: 1633.30\nLoglikelihood: 2280.31\nR-squared: NaN\nWald: 0.09 ∼ F(2, 531) ⟹ Pr > F = 0.9116\nFormula: CRMRTE ~ 1 + PrbConv + (AvgSen ~ PrbPris) + absorb(County + Year)\nVariance Covariance Estimator: OIM\n────────────────────────────────────────────────────────────────────────────────────\n                   PE           SE      t-value  Pr > |t|         2.50%       97.50%\n────────────────────────────────────────────────────────────────────────────────────\n(Intercept)  0.027409     0.0208187    1.31656     0.1886  -0.0134881    0.0683061  \nPrbConv      6.00404e-5   0.000214966  0.279301    0.7801  -0.000362248  0.000482329\nAvgSen       0.000462028  0.0023309    0.198219    0.8429  -0.00411688   0.00504094 \n────────────────────────────────────────────────────────────────────────────────────"
    	# IV Random
    	model = fit(RandomEffectsEstimator,
                	@formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
                	data,
					panel = :County,
					time = :Year)
		@test sprint(show, model) == "One-way Random Effect Model\nLongitudinal dataset: County, Year\nBalanced dataset with 90 panels of length 7\nindividual error component: 0.0413\nidiosyncratic error component: 0.0074\nρ: 0.9691\nNumber of observations: 630\nNull Loglikelihood: 2268.01\nLoglikelihood: 2248.34\nR-squared: NaN\nWald: 0.03 ∼ F(2, 626) ⟹ Pr > F = 0.9671\nFormula: CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)\nVariance Covariance Estimator: OIM\n───────────────────────────────────────────────────────────────────────────────────────\n                    PE           SE        t-value  Pr > |t|         2.50%       97.50%\n───────────────────────────────────────────────────────────────────────────────────────\n(Intercept)   0.037747     0.0241684     1.56183      0.1188  -0.00971405   0.085208   \nPrbConv       1.39581e-5   0.000200244   0.0697054    0.9445  -0.000379273  0.000407189\nAvgSen       -0.000688924  0.00266514   -0.258494     0.7961  -0.00592262   0.00454478 \n───────────────────────────────────────────────────────────────────────────────────────"
  	end
	@testset "Heteroscedasticity Consistent Variance Covariance Estimators" begin
		data = dataset("Ecdat", "Crime")
    	# Pooling
    	model = fit(EconometricModel,
                	@formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
                	data)
		@test vcov(model) ≈ vcov(model, OIM)
		@test vcov(model, HC0) ≈ [ 1.51708E-05  3.58518E-07 -7.30083E-07 -2.03240E-05
									 3.58518E-07  2.85571E-07 -3.58783E-08 -4.13434E-07
									-7.30083E-07 -3.58783E-08  9.36232E-08 -1.46277E-07
									-2.03240E-05 -4.13434E-07 -1.46277E-07  5.17390E-05
								   ] rtol = 1e-6
		@test vcov(model, HC1) ≈ [ 1.52677E-05  3.60808E-07 -7.34748E-07 -2.04539E-05
		   							 3.60808E-07  2.87396E-07 -3.61076E-08 -4.16076E-07
									-7.34748E-07 -3.61076E-08  9.42214E-08 -1.47212E-07
									-2.04539E-05 -4.16076E-07 -1.47212E-07  5.20696E-05
								   ] rtol = 1e-6
		@test vcov(model, HC2) ≈ [ 1.56208E-05  9.05631E-08 -7.36000E-07 -2.08761E-05
									 9.05631E-08  6.78782E-07 -5.03354E-08 -3.93296E-08
									-7.36000E-07 -5.03354E-08  9.59715E-08 -1.59383E-07
									-2.08761E-05 -3.93296E-08 -1.59383E-07  5.27586E-05
								   ] rtol = 1e-6
		@test vcov(model, HC3) ≈ [ 1.66839E-05 -9.88963E-07 -7.13420E-07 -2.23048E-05
									-9.88963E-07  2.16675E-06 -1.04165E-07  1.50077E-06
									-7.13420E-07 -1.04165E-07  9.98145E-08 -2.14382E-07
									-2.23048E-05  1.50077E-06 -2.14382E-07  5.50463E-05
								   ] rtol = 1e-6
		@test vcov(model, HC4) ≈ [ 3.21597E-05 -2.14119E-05 -4.18897E-09 -4.43245E-05
									-2.14119E-05  2.97197E-05 -1.09486E-06  3.08355E-05
									-4.18897E-09 -1.09486E-06  1.38192E-07 -1.26498E-06
									-4.43245E-05  3.08355E-05 -1.26498E-06  8.67924E-05
								   ] rtol = 1e-6
		dv = fit(EconometricModel,
				 @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + County),
				 data,
				 contrasts = Dict(:County => DummyCoding()))
		fe = fit(EconometricModel,
				 @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County)),
				 data)
		@test vcov(fe)[2:end,2:end] ≈ vcov(dv)[2:4, 2:4]
		@test vcov(fe, HC1)[2:end,2:end] ≈ vcov(dv, HC1)[2:4, 2:4]
	end
end
# Nominal Models
@testset "Nominal Models" begin
	data = joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |>
		CSV.read |>
		(data -> select(data, [:insure, :age, :male, :nonwhite, :site])) |>
		dropmissing |>
		(data -> categorical!(data, [:insure, :site]))
	model = fit(EconometricModel,
				@formula(insure ~ age + male + nonwhite + site),
				data,
				contrasts = Dict(:insure => DummyCoding(base = "Uninsure")))
	@test sprint(show, model) == "Probability Model for Nominal Response\nCategories: Uninsure, Indemnity, Prepaid\nNumber of observations: 615\nNull Loglikelihood: -555.85\nLoglikelihood: -534.36\nR-squared: 0.0387\nLR Test: 42.99 ∼ χ²(10) ⟹ Pr > χ² = 0.0000\nFormula: insure ~ 1 + age + male + nonwhite + site\n───────────────────────────────────────────────────────────────────────────────────────────────────\n                                       PE         SE       t-value  Pr > |t|       2.50%     97.50%\n───────────────────────────────────────────────────────────────────────────────────────────────────\ninsure: Indemnity ~ (Intercept)   1.28694     0.59232     2.17271     0.0302   0.123689   2.4502   \ninsure: Indemnity ~ age           0.00779612  0.0114418   0.681372    0.4959  -0.0146743  0.0302666\ninsure: Indemnity ~ male         -0.451848    0.367486   -1.22957     0.2193  -1.17355    0.269855 \ninsure: Indemnity ~ nonwhite     -0.217059    0.425636   -0.509965    0.6103  -1.05296    0.618843 \ninsure: Indemnity ~ site: 2       1.21152     0.470506    2.57493     0.0103   0.287497   2.13554  \ninsure: Indemnity ~ site: 3       0.207813    0.366293    0.56734     0.5707  -0.511547   0.927172 \ninsure: Prepaid ~ (Intercept)     1.55666     0.596327    2.61041     0.0093   0.385533   2.72778  \ninsure: Prepaid ~ age            -0.00394887  0.0115993  -0.340439    0.7336  -0.0267287  0.018831 \ninsure: Prepaid ~ male            0.109846    0.365187    0.300793    0.7637  -0.607343   0.827035 \ninsure: Prepaid ~ nonwhite        0.757718    0.419575    1.80592     0.0714  -0.0662835  1.58172  \ninsure: Prepaid ~ site: 2         1.32456     0.469789    2.81947     0.0050   0.401941   2.24717  \ninsure: Prepaid ~ site: 3        -0.380175    0.372819   -1.01973     0.3083  -1.11235    0.352001 \n───────────────────────────────────────────────────────────────────────────────────────────────────"
	β, V, σ = coef(model), vcov(model), stderror(model)
	@test β ≈ [1.286943, 0.0077961, -0.4518496, -0.2170589, 1.211563, 0.2078123,
               1.556656, -0.0039489, 0.1098438, 0.7577178, 1.324599, -0.3801756
              ] rtol = 1e-3
  	@test V ≈ [ 0.35084518 -0.00590456 -0.02519209 -0.01837512 -0.08995794 -0.08264268  0.29928937 -0.00510107 -0.02257270 -0.01585492 -0.07392512 -0.06808314
               -0.00590456  0.00013092 -0.00046225 -0.00028066  0.00041004  0.00040470 -0.00509454  0.00011354 -0.00039387 -0.00024122  0.00033847  0.00035236
               -0.02519209 -0.00046225  0.13504644  0.01565749  0.01009559  0.00933967 -0.02288847 -0.00039497  0.11365141  0.01369872  0.00945927  0.00802593
               -0.01837512 -0.00590456 -0.02519209  0.18116609  0.01886597 -0.02821348 -0.01638629 -0.00023179  0.01314666  0.15068111  0.01664999 -0.02345786
               -0.08995794 -0.00590456 -0.02519209 -0.01837512  0.22138224 0.064050660 -0.07438947  0.00035751  0.00845363  0.01594691  0.19895500  0.05189742
               -0.08264268  0.00040470  0.00933967 -0.02821348  0.06405066 0.134170290 -0.06796097  0.00034828  0.00826890 -0.02353016  0.05182727  0.11060487
                0.29928937 -0.00509454 -0.02288847 -0.01638629 -0.07438947 -0.06796097  0.35560782 -0.00601966 -0.02683161 -0.02105510 -0.08991231 -0.08165509
               -0.00510107  0.00011354 -0.00039497 -0.00023179  0.00035751  0.00034828 -0.00601966  0.00013455 -0.00047223 -0.00028680  0.00039990  0.0004179
               -0.02257270 -0.00039387  0.11365141  0.01314666  0.00845363  0.00826890 -0.02683161 -0.00047223  0.13336253  0.01625685  0.01101870  0.00922333
               -0.01585492 -0.00024122  0.01369872  0.15068111  0.01594691 -0.02353016 -0.02105510 -0.00028680  0.01625685  0.17604391  0.02099458 -0.03008391
               -0.07392512  0.00033847  0.00945927  0.01664999  0.19895500  0.05182727 -0.08991231  0.00039990  0.01101870  0.02099458  0.22070774  0.06273376
               -0.06808314  0.00035236  0.00802593 -0.02345786  0.05189742  0.11060487 -0.08165509  0.00041790  0.00922333 -0.03008391  0.06273376  0.13899387
              ] |> Hermitian rtol = 1e-3
	@test σ ≈ [0.5923219, 0.0114418, 0.3674867, 0.4256361, 0.4705127, 0.3662926,
  			   0.5963286, 0.0115994, 0.3651883, 0.4195759, 0.4697954, 0.3728188
			  ] rtol = 1e-4
end
# Ordinal Models
@testset "Ordinal Models" begin
	data = dataset("Ecdat", "Kakadu") |>
		(data -> select(data, [:RecParks, :Sex, :Age, :Schooling]))
	data.RecParks = convert(Vector{Int}, data.RecParks)
	data.RecParks = levels!(categorical(data.RecParks, ordered = true), collect(1:5))
	model = fit(EconometricModel,
              	@formula(RecParks ~ Age + Sex + Schooling),
              	data)
  	@test sprint(show, model) == "Probability Model for Ordinal Response\nCategories: 1 < 2 < 3 < 4 < 5\nNumber of observations: 1827\nNull Loglikelihood: -2677.60\nLoglikelihood: -2657.40\nR-squared: 0.0075\nLR Test: 40.42 ∼ χ²(3) ⟹ Pr > χ² = 0.0000\nFormula: RecParks ~ Age + Sex + Schooling\n──────────────────────────────────────────────────────────────────────────────────────────\n                          PE          SE        t-value  Pr > |t|        2.50%      97.50%\n──────────────────────────────────────────────────────────────────────────────────────────\nAge                  0.00943647  0.00254031    3.71469     0.0002   0.00445424   0.0144187\nSex: male           -0.0151659   0.0846365    -0.179188    0.8578  -0.181161     0.150829 \nSchooling           -0.103902    0.0248742    -4.17711     <1e-4   -0.152687    -0.0551174\n(Intercept): 1 | 2  -2.92405     0.191927    -15.2352      <1e-48  -3.30047     -2.54763  \n(Intercept): 2 | 3  -1.54922     0.171444     -9.03632     <1e-18  -1.88547     -1.21297  \n(Intercept): 3 | 4  -0.298938    0.166904     -1.79108     0.0734  -0.626281     0.0284051\n(Intercept): 4 | 5   0.669835    0.167622      3.9961      <1e-4    0.341083     0.998587 \n──────────────────────────────────────────────────────────────────────────────────────────"
  	β, V, σ = coef(model), vcov(model), stderror(model)
  	@test β ≈ [ 0.009437926, -0.015143049, -0.103911316,
               -2.92391240,  -1.549266200, -0.298963900, 0.6698249] rtol = 1e-4
	@test V ≈ [ 0.00000646 -0.0000007000  0.000013250 0.00031473	0.0003161358 0.000321276 0.0003270734
               -0.00000070  0.0071633800 -0.000184930 0.00294759	0.0029647442 0.002995366 0.0029904485
                0.00001325 -0.0001849300  0.000618730 0.00288365	0.0028548209 0.002791361 0.0027318928
                0.00031473  0.0029475900  0.002883650 0.03684071	0.0289239290 0.026986237 0.026498765
                0.00031614  0.0029647400  0.002854820 0.02892393	0.0293971651 0.027160433 0.0266006675
                0.000321276 0.0029953660  0.002791361 0.02698623	0.0271604331 0.027860143 0.0270590982
                0.0003270734 0.002990448  0.002731893 0.02649876	0.0266006675 0.027059098 0.0281002038
              ] rtol = 1e-3
    @test σ ≈ [0.002540789, 0.084636770, 0.024874381,
               0.191939349, 0.171456015, 0.166913580, 0.167631154] rtol = 1e-4
  	data = joinpath(dirname(pathof(Econometrics)), "..", "data", "auto.csv") |>
		CSV.read |>
    	(data -> select(data, [:rep77, :foreign, :length, :mpg])) |>
    	dropmissing
  	data.rep77 = levels!(categorical(data.rep77; ordered = true),
                       	 ["Poor", "Fair", "Average", "Good", "Excellent"])
  	model = fit(EconometricModel,
              	@formula(rep77 ~ foreign + length + mpg),
              	data)
  	@test sprint(show, model) == "Probability Model for Ordinal Response\nCategories: Poor < Fair < Average < Good < Excellent\nNumber of observations: 66\nNull Loglikelihood: -89.90\nLoglikelihood: -78.25\nR-squared: 0.1295\nLR Test: 23.29 ∼ χ²(3) ⟹ Pr > χ² = 0.0000\nFormula: rep77 ~ foreign + length + mpg\n─────────────────────────────────────────────────────────────────────────────────────────────\n                                    PE         SE     t-value  Pr > |t|       2.50%    97.50%\n─────────────────────────────────────────────────────────────────────────────────────────────\nforeign: Foreign                2.89681    0.790641   3.66387    0.0005   1.31684     4.47678\nlength                          0.0828275  0.02272    3.64558    0.0005   0.0374253   0.12823\nmpg                             0.230768   0.0704548  3.2754     0.0017   0.0899749   0.37156\n(Intercept): Poor | Fair       17.9275     5.55119    3.22948    0.0020   6.83431    29.0206 \n(Intercept): Fair | Average    19.8651     5.59648    3.54956    0.0007   8.68139    31.0487 \n(Intercept): Average | Good    22.1033     5.70894    3.8717     0.0003  10.6949     33.5117 \n(Intercept): Good | Excellent  24.6921     5.89075    4.19168    <1e-4   12.9204     36.4639 \n─────────────────────────────────────────────────────────────────────────────────────────────"
  	β, V, σ = coef(model), vcov(model), stderror(model)
  	@test β ≈ [2.89679875, 0.08282676, 0.23076532, 17.92728, 19.86486, 22.10311, 24.69193] atol = 1e-3
  	@test V ≈ [ 0.62578335	0.0111044038 0.012321411 2.4421745	2.4836507	2.5750058	2.6957587
             	0.0111044	0.0005186744 0.001222385 0.1243671	0.1258979	0.1284727	0.1319772
                0.01232141	0.0012223847 0.004974678 0.3328169	0.3368057	0.3430257	0.3549711
                2.44217451	0.1243670884 0.332816913 30.9544666	31.0279194	31.6005528	32.5099187
                2.48365067	0.1258979463 0.336805731 31.0279194	31.4396158	31.9986912	32.9170873
                2.57500582	0.1284726666 0.343025653 31.6005528	31.9986912	32.7044311	33.6189243
                2.69575871	0.1319772302 0.354971081 32.5099187	32.9170873	33.6189243	34.8533724
              ] rtol = 1e-2
    @test σ ≈ [0.79106469, 0.02277442, 0.07053140,
               5.56367384, 5.60710405, 5.71877881, 5.90367448] rtol = 1e-2
end
