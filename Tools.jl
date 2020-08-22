# Do not need a module here
# Useful or Unuseful tools
# module Tools

# a very stupid package
using GeometricalPredicates

using PyPlot
# using DelimitedFiles for Julia 1.0

# export TriInterp, Tribasisfunc, plot2dBC

# bilinear basis function
# No.k nodal basis, k = 1, 2, 3, 4, check!
# 4--3
# 1--2
function BilinearBasisFunc(x1, x2, x3, x4, x, k)
    if k == 1 
        return (x3[1] - x[1, :]) .* (x[2, :] - x3[2]) / ( (x3[1] - x1[1]) * (x1[2] - x3[2]) )
    end
    if k == 2
        return (x[1, :] - x4[1]) .* (x[2, :] - x4[2]) / ( (x2[1] - x4[1]) * (x2[2] - x4[2]) )
    end
    if k == 3
        return (x[1, :] - x1[1]) .* (x1[2] - x[2, :]) / ( (x3[1] - x1[1]) * (x1[2] - x3[2]) )
    end
    if k == 4
        return (x2[1] - x[1, :]) .* (x2[2] - x[2, :]) / ( (x2[1] - x4[1]) * (x2[2] - x4[2]) )
    end
end   

# Interpolation by using basis function in one element using Vectorization
# Much faster than normal
# xx can be all atoms in the element
# return w should be weight matrix
function BilinearInterpVec(x, u, xx)
    x1 = x[:, 1]; x2 = x[:, 2]; x3 = x[:, 3]; x4 = x[:, 4];
    n = size(xx, 2)
    uu = zeros(2, n)
    ux = u[1, :]; uy = u[2, :];
    w = zeros(n, 4)
    for k = 1:4
        w[:, k] = BilinearBasisFunc(x1, x2, x3, x4, xx, k)
        uu[1, :] += w[:, k] * ux[k]
        uu[2, :] += w[:, k] * uy[k]
    end
    return uu, w
end

# for bilinear grid
# maybe useless since mesh information is from deal ii
function element(H::Float64) 
   
    N = Int64(R / H + 1) 
    n = N - 1
    # allocate the elements
    T = zeros(Int64, 4, n^2)   
    idx = 0
    for icol = 1:n, irow = 1:n
        a = (icol-1)*(n+1)+irow
        idx += 1
        T[:,idx] = [a; a+1; a+n+2; a+n+1]  
    end
    
    t = linspace(0.0, R, N)
    O = ones(N)
    X = A * [(t * O')[:]'; (O * t')[:]']
    Xvecs = [[(t * O')[:]'; (O * t')[:]']; zeros(N^2)'] |> vecs
    
    return X, Xvecs, T
    
end

# normolized the position to [1.0, 2.0]
@inline function mymap(X::Array{Float64,2})
     x = X[1, :]; y = X[2, :]
     xmax = maximum(x); ymax = maximum(y)
     x = 1.5 + x ./ (2*xmax)
     y = 1.5 + y ./ (2*ymax)  # 
     return hcat(x, y)'
end

# a very inefficient implementation
# maybe need inline function
function Intriangleineff(xa, T)
    ind = []
    xamap = mymap(xa)
    # write it as inline function
    xaPoint = xapoint(xamap)
    tic()
    @simd for iT = 1:size(T, 2)
        @inbounds xTpre = mymap(X[:, T[:, iT]])
        xT = consxT(xTpre)
        mytri = Primitive(xT[1], xT[2], xT[3])
        @inbounds i = [intriangle(mytri, xaPoint[i]) for i in 1:size(xamap, 2)]
        #@inbounds push!(ind, [intriangle(mytri, xaPoint[i]) for i in 1:size(xamap, 2)])
    end
    toc()

end

@inline xapoint(xamap::Array{Float64,2}) = [Point(xamap[1, i], xamap[2, i]) for i in 1:size(xamap, 2)]

@inline consxT(xTpre::Array{Float64,2}) = [Point(xTpre[1, 1], xTpre[2, 1]); Point(xTpre[1, 2], xTpre[2, 2]); 
                                           Point(xTpre[1, 3], xTpre[2, 3])]

@inline checkPoint(mytri, xa) = [intriangle(mytri, xaPoint[i]) for i in 1:size(xamap, 2)]

# inefficeient, calculate all 3 and then choose
@inline function Tribasisfunc(xT, x)
    # No.k nodal basis of triangulate interpolation
    # k = 1, 2, 3
    # xT should be a 2*3 matrix

    A = [ones(1, 3); xT]' \ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    M = hcat(ones(size(x, 2), 1), x') * A
    
    return M # N * 3 matrix

end  

@inline function TriInterp(x, u, xx)
    # Triangulate interpolation using basis function in one element
    # xx can be all atoms in the element
    # return w which should be the weighted matrix
    # x should be a 2*3 matrix
    uu = zeros(2, size(xx, 2))
    ux = u[1, :]; uy = u[2, :];
    w = zeros(size(xx, 2), 3)
    M = Tribasisfunc(x, xx)
    @simd for k = 1:3      
        w[:, k] = M[:, k]
        @inbounds uu[1, :] += w[:, k] * ux[k]
        @inbounds uu[2, :] += w[:, k] * uy[k]
    end
    return uu, w
end


# reference bilinear basis function
# 4-3
# 1-2
ϕ_1(ξ, η) = (1 - ξ) .* (1 - η)
ϕ_2(ξ, η) = ξ .* (1 - η)
ϕ_3(ξ, η) = ξ .* η
ϕ_4(ξ, η) = (1 - ξ) .* η

function biInterp(x::Array{Float64, 2}, U::Array{Float64, 2})
    # w should be 4*N
#     w = hcat((1 - x[1, :]).*(1 - x[2, :]), 
#              x[1, :] .* (1 - x[2, :]), 
#              x[1, :] .* x[2, :],
#              (1 - x[1, :]) .* x[2, :])'
    ξ, η = x[1, :], x[2, :] 
    w = hcat(ϕ_1(ξ, η), ϕ_2(ξ, η), 
             ϕ_3(ξ, η), ϕ_4(ξ, η))'
    u = U * w
    return w, u
end

function para2square(X::Array{Float64,2}, x)
    # transformation from T to Tref
    # ξ = a1 + a2 * x + a3 * y + a4 * x * y
    # η = b1 + b2 * x + b3 * y + b4 * x * y
    # X = [0 1 1.5 0.5; 0 0 √3/2 √3/2] should be 2*4 matrix
    # x = [1 0] should be 2*N matrix
    A = [1 X[1, 1] X[2, 1] X[1, 1]*X[2, 1] 0 0 0 0;
         1 X[1, 2] X[2, 2] X[1, 2]*X[2, 2] 0 0 0 0;
         1 X[1, 3] X[2, 3] X[1, 3]*X[2, 3] 0 0 0 0;
         1 X[1, 4] X[2, 4] X[1, 4]*X[2, 4] 0 0 0 0;
         0 0 0 0 1 X[1, 1] X[2, 1] X[1, 1]*X[2, 1];
         0 0 0 0 1 X[1, 2] X[2, 2] X[1, 2]*X[2, 2];
         0 0 0 0 1 X[1, 3] X[2, 3] X[1, 3]*X[2, 3];
         0 0 0 0 1 X[1, 4] X[2, 4] X[1, 4]*X[2, 4];]
    b = [0; 1; 1; 0; 0; 0; 1; 1;]
    a = A \ b
    x = reshape(a, 4, 2)' * hcat(ones(size(x, 2)), x[1, :], x[2, :], x[1, :].*x[2, :])'
end


# reference trilinear basis function
#   8-7
# 5-6
#   4-3
# 1-2
ϕ3_1(x, y, z) = 1/8 * (1 - x) .* (1 - y) .* (1 - z)
ϕ3_2(x, y, z) = 1/8 * (1 + x) .* (1 - y) .* (1 - z)
ϕ3_3(x, y, z) = 1/8 * (1 + x) .* (1 + y) .* (1 - z)
ϕ3_4(x, y, z) = 1/8 * (1 - x) .* (1 + y) .* (1 - z)
ϕ3_5(x, y, z) = 1/8 * (1 - x) .* (1 - y) .* (1 + z)
ϕ3_6(x, y, z) = 1/8 * (1 + x) .* (1 - y) .* (1 + z)
ϕ3_7(x, y, z) = 1/8 * (1 + x) .* (1 + y) .* (1 + z)
ϕ3_8(x, y, z) = 1/8 * (1 - x) .* (1 + y) .* (1 + z)

function triInterp(x::Array{Float64, 2}, U::Array{Float64, 2})
    # w should be 8*N
    x1, x2, x3 = x[1, :], x[2, :], x[3, :] 
    w = hcat(ϕ3_1(x1, x2, x3), ϕ3_2(x1, x2, x3), ϕ3_3(x1, x2, x3), ϕ3_4(x1, x2, x3),
             ϕ3_5(x1, x2, x3), ϕ3_6(x1, x2, x3), ϕ3_7(x1, x2, x3), ϕ3_8(x1, x2, x3))'
    u = U * w
    return w, u
end

function hex2cube(X::Array{Float64,2}, x)
    # transformation from T to Tref
    # ξ = a1 + a2*x + a3*y + a4*z + a5*x*y + a6*x*z + a7*y*z + a8*x*y*z
    # η = b1 + b2*x + b3*y + b4*z + b5*x*y + b6*x*z + b7*y*z + b8*x*y*z
    # γ = c1 + c2*x + c3*y + c4*z + c5*x*y + c6*x*z + c7*y*z + c8*x*y*z
    # X should be 3*8 matrix
    # x should be 2*N matrix
    A1 = zeros(size(X, 2), size(X, 2))
    O = copy(A1)
    for i = 1:size(X, 2)
        A1[i, :] = [1 X[1, i] X[2, i] X[3, i] X[1, i]*X[2, i] X[1, i]*X[3, i] X[2, i]*X[3, i] X[1, i]*X[2, i]*X[3, i] ]
    end
    A = [A1 O O; O A1 O; O O A1]
    b = [-1; 1; 1; -1; -1; 1; 1; -1;
         -1; -1; 1; 1; -1; -1; 1; 1;
         -1; -1; -1; -1; 1; 1; 1; 1;] # should be 24*1
    a = A \ b
    x = reshape(a, 8, 3)' * hcat(ones(size(x, 2)), x[1, :], x[2, :], x[3, :], x[1, :].*x[2, :], x[1, :].*x[3, :], x[2, :].*x[3, :], x[1, :].*x[2, :].*x[3, :])'
end



#####################################################################################################

function plot2dBC(at::JuLIP.Atoms{Float64, Int64}; iBC=nothing)
    
    x, y, _ = xyz(at)
    plot(x, y, "ro", markersize=6) 
    if iBC != nothing
        plot(x[iBC], y[iBC], "go", markersize=6)
    end
    axis("equal")
    
end 

function plot3dBC(at; iBC=nothing)
    
    X = positions(at) |> mat 
    plot3D(X[1,:], X[2,:], X[3, :], "ro", markersize=4)
    
    if iBC != nothing
        l = copy(iBC)
        o = [l/2, l/2, l/2]
        I = []
        for i = 1:size(X, 2)
            x = X[:, i]
            if abs(x[1]-o[1])>l/2-1.8 || abs(x[2]-o[2])>l/2-1.8 || abs(x[3]-o[3])>l/2-1.8
                push!(I, i)
            end
        end
        plot3D(X[1,I], X[2,I], X[3, I], "go", markersize=4)
    end
    
end 

# Only for square mesh
# maybe useful
# nodes in the fine mesh position
function idxElement2Atom(X, T, xref, t::Int64)
    idxint = Int64[]; idx = Int64[]
    vertex = X[:, T[:, t]]
    v1 = vertex[:, 1]; v2 = vertex[:, 3];
    h = v2[1] - v1[1];
    midpoint = [0.5*(v1[1]+v2[1]), 0.5*(v1[2]+v2[2])]
    iint = find((abs.(midpoint[1] .- xref[1, :]) .< h/2) .& (abs.(midpoint[2] .- xref[2, :]) .< h/2))
    i = find((abs.(midpoint[1] .- xref[1, :]) .<= h/2) .& (abs.(midpoint[2] .- xref[2, :]) .<= h/2))
    push!(idxint, iint...)
    push!(idx, i...)
    return idxint, idx 
end

# end