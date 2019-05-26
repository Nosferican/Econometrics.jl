using Test, Econometrics, CSV, RDatasets, LinearAlgebra

# Linear Models
@testset "Linear Models" begin
	@testset "Balanced Panel Data" begin
    # Balanced Panel Data
    data = dataset("Ecdat", "Crime")
    # Pooling
    model = fit(EconometricModel,
                @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris),
                data)
    @test nullloglikelihood(model) ≈ 1633.305 rtol = 1e-3
    @test loglikelihood(model) ≈ 1643.304 rtol = 1e-3
    @test dof(model) == 5
    # This corresponds to BIC from R's lm, Stata uses rank(vcov(model))
    @test bic(model) ≈ -3254.379 rtol = 1e-3
    @test coef(model) ≈ [0.0186413, -0.0011647, 0.0002362, 0.0273394] rtol = 1e-5
    @test vcov(model) ≈ [  1.858177e-05 -1.620480e-07 -6.464872e-07 -2.860941e-05
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
    model = fit(EconometricModel,
                @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + between(County)),
                data)
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
    #
    @test_broken nullloglikelihood(model) ≈ 2277.488 rtol = 1e-3
    @test_broken loglikelihood(model) ≈ 1643.304 rtol = 1e-3
    xtreg_fe_i = fit(EconometricModel,
                     @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + absorb(County + Year)),
                     data)
    # Random Effects
    xtreg = fit(EconometricModel,
                @formula(CRMRTE ~ PrbConv + AvgSen + PrbPris + PID(County) + TID(Year)),
                data)
    # IV Pooling
    ivreg = fit(EconometricModel,
                @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris)),
                data)
    # IV Between
    ivreg_be = fit(EconometricModel,
                   @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + between(County)),
                   data)
    # IV Within
    ivreg_fe = fit(EconometricModel,
                   @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County)),
                   data)
    ivreg_fe_i = fit(EconometricModel,
                     @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + absorb(County + Year)),
                     data)
    # IV Random
    ivreg = fit(EconometricModel,
                @formula(CRMRTE ~ PrbConv + (AvgSen ~ PrbPris) + PID(County) + TID(Year)),
                data)
  end
end
# Nominal Models
@testset "Nominal Models" begin
  data = joinpath(dirname(pathof(Econometrics)), "..", "data", "insure.csv") |>
	CSV.read |>
    (data -> data[[:insure, :age, :male, :nonwhite, :site]]) |>
    (data -> dropmissing!(data, disallowmissing = true)) |>
    (data -> categorical!(data, [:insure, :site]))
  model = fit(EconometricModel,
              @formula(insure ~ 1 + age + male + nonwhite + site),
              data,
              contrasts = Dict(:insure => DummyCoding(base = "Uninsure")))
  β, V = coef(model), vcov(model)
  @test β ≈ [1.286943, 0.0077961, -0.4518496, -0.2170589, 1.211563, 0.2078123,
             1.556656, -0.0039489, 0.1098438, 0.7577178, 1.324599, -0.3801756
             ] rtol = 1e-3
  @test V ≈ Hermitian(
  			[ 0.35084518 -0.00590456 -0.02519209 -0.01837512 -0.08995794 -0.08264268  0.29928937 -0.00510107 -0.02257270 -0.01585492 -0.07392512 -0.06808314
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
            ]) rtol = 1e-3
  @test sqrt.(diag(V)) ≈ [0.5923219, 0.0114418, 0.3674867, 0.4256361, 0.4705127, 0.3662926,
  						  0.5963286, 0.0115994, 0.3651883, 0.4195759, 0.4697954, 0.3728188
						 ] rtol = 1e-4
end
# Ordinal Models
@testset "Ordinal Models" begin
  data = dataset("Ecdat", "Kakadu")[[:RecParks, :Sex, :Age, :Schooling]]
  data.RecParks = convert(Vector{Int}, data.RecParks)
  data.RecParks = levels!(categorical(data.RecParks, ordered = true), collect(1:5))
  model = fit(EconometricModel,
               @formula(RecParks ~ Age + Sex + Schooling),
               data,
               contrasts = Dict(:RecParks => DummyCoding(levels = collect(1:5)))
               )
  β, V = coef(model), vcov(model)
  @test β ≈ [0.009437926, -0.015143049, -0.103911316,
             -2.92391240, -1.549266200, -0.298963900, 0.6698249] rtol = 1e-4
  @test V ≈ [ 0.00000646 -0.0000007000  0.000013250 0.00031473	0.0003161358 0.000321276 0.0003270734
             -0.00000070  0.0071633800 -0.000184930 0.00294759	0.0029647442 0.002995366 0.0029904485
              0.00001325 -0.0001849300  0.000618730 0.00288365	0.0028548209 0.002791361 0.0027318928
              0.00031473  0.0029475900  0.002883650 0.03684071	0.0289239290 0.026986237 0.026498765
              0.00031614  0.0029647400  0.002854820 0.02892393	0.0293971651 0.027160433 0.0266006675
              0.000321276 0.0029953660  0.002791361 0.02698623	0.0271604331 0.027860143 0.0270590982
              0.0003270734 0.002990448  0.002731893 0.02649876	0.0266006675 0.027059098 0.0281002038
            ] rtol = 1e-3
  @test sqrt.(diag(V)) ≈ [0.002540789, 0.084636770, 0.024874381,
                          0.191939349, 0.171456015, 0.166913580, 0.167631154] rtol = 1e-4
  data = joinpath(dirname(pathof(Econometrics)), "..", "data", "auto.csv") |>
	CSV.read |>
    (data -> data[[:rep77, :foreign, :length, :mpg]]) |>
    (data -> dropmissing!(data, disallowmissing = true))
  data.rep77 = levels!(categorical(data.rep77; ordered = true),
                       ["Poor", "Fair", "Average", "Good", "Excellent"])
  model = fit(EconometricModel,
              @formula(rep77 ~ foreign + length + mpg),
              data)
  β, V = coef(model), vcov(model)
  @test β ≈ [2.89679875, 0.08282676, 0.23076532, 17.92728, 19.86486, 22.10311, 24.69193] atol = 1e-3
  @test V ≈ [0.62578335	0.0111044038 0.012321411 2.4421745	2.4836507	2.5750058	2.6957587
             0.0111044	0.0005186744 0.001222385 0.1243671	0.1258979	0.1284727	0.1319772
             0.01232141	0.0012223847 0.004974678 0.3328169	0.3368057	0.3430257	0.3549711
             2.44217451	0.1243670884 0.332816913 30.9544666	31.0279194	31.6005528	32.5099187
             2.48365067	0.1258979463 0.336805731 31.0279194	31.4396158	31.9986912	32.9170873
             2.57500582	0.1284726666 0.343025653 31.6005528	31.9986912	32.7044311	33.6189243
             2.69575871	0.1319772302 0.354971081 32.5099187	32.9170873	33.6189243	34.8533724
            ] rtol = 1e-2
  @test sqrt.(diag(V)) ≈ [0.79106469, 0.02277442, 0.07053140,
                          5.56367384, 5.60710405, 5.71877881, 5.90367448] rtol = 1e-2
end
