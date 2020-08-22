#####################################################################################################

"""
standard isotropic CLE edge dislocation solution
"""
function ulin_edge_isotropic(X, b, ν)
    x, y = X[1,:], X[2,:]
    r² = x.^2 + y.^2
    ux = b/(2*π) * ( angle.(x + im*y) + (x .* y) ./ (2*(1-ν) * r²) )
    uy = -b/(2*π) * ( (1-2*ν)/(4*(1-ν)) * log.(r²) + - 2 * y.^2 ./ (4*(1-ν) * r²) )
    return [ux'; uy']
end

"""
lattice corrector to CLE edge solution; cf EOS paper
"""
function xi_solver(Y::Vector, b; TOL = 1e-10, maxnit = 5)
   ξ1(x::Real, y::Real, b) = x - b * angle.(x + im * y) / (2*π)
   dξ1(x::Real, y::Real, b) = 1 + b * y / (x^2 + y^2) / (2*π)
    y = Y[2]
    x = y
    for n = 1:maxnit
        f = ξ1(x, y, b) - Y[1]
        if abs(f) <= TOL; break; end
        x = x - f / dξ1(x, y, b)
    end
    if abs(ξ1(x, y, b) - Y[1]) > TOL
        warn("newton solver did not converge at Y = $Y; returning input")
        return Y
    end
    return [x, y]
end

"""
EOSShapeev edge dislocation solution
"""
function ulin_edge_eos(X, b, ν)
    Xmod = zeros(2, size(X, 2))
    for n = 1:size(X,2)
        Xmod[:, n] = xi_solver(X[1:2,n], b)
    end
    return ulin_edge_isotropic(Xmod, b, ν)
end

function edge_predictor!(at::AbstractAtoms; b = 1.0, xicorr = true, ν = 0.25)
   X = positions(at) |> mat
   if xicorr
      X[1:2,:] += ulin_edge_eos(X, b, ν)
   else
      X[1:2,:] += ulin_edge_isotropic(X, b, ν)
   end
   set_positions!(at, X)
   return at
end

#####################################################################################################