"""
    wald(obj::EconometricModel, vce::VCE = obj.vce)

Provides the Wald statistic as Wald value, a F-Distribution, and the associated p-value.
"""
function wald(obj::EconometricModel, vce::VCE = obj.vce)
    β = coef(obj)
    V = vcov(obj, vce)
    k = length(β) - 1
    rdf = dof_residual(obj)
    R = hasintercept(obj) ? hcat(zeros(k), I) : I
    Bread = R * β
    Meat = inv(bunchkaufman!(Hermitian(R * V * R')))
    W = (Bread' * Meat * Bread) / size(R, 1)
    if size(R, 1) > 0
        F = FDist(size(R, 1), rdf)
        p = ccdf(F, W)
    else
        F = FDist(1, 1)
        p = NaN
    end
    W, F, p
end
