@testset "Issues" begin
    @testset "Nonpositive Weights" begin
        x = randn(100)
        y = 2*x + randn(100)
        w = randn(100)
        data = DataFrame(x = x, y = y, w = w)
        # Test that no error is thrown
        # https://github.com/JuliaLang/julia/issues/18780#issuecomment-251534863
        @test fit(EconometricModel, @formula(y ~ x), data; wts = :w) isa Any
    end
end
