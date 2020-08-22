# Construct the configuration
# A: non-singular matrix
# L: size, default layer = 0 means no DBC 
# vacancies position
# Author: Yangshuai Wang, Date: 11/26/2018

using JuLIP

function getX(A::Array{Float64, 2}, L::Int64, vacancies, shape::Symbol; layer=0, task="2DPoint")

    # 六边形结构
    if shape == :hexagon
        X, iBC = hexregion(L; layer=layer)
    end
    # 正方形结构
    if shape == :square
        X, iBC = squregion(L; layer=layer)
    end
    # 平行四边形结构
    if shape == :parallel
        X, iBC = parregion(A, L, vacancies=vacancies, layer=layer, task=task)
    end
    # 圆形结构
    if shape == :ball
        X, iBC = balregion(L; layer=layer)
    end    
    return X, iBC
    
end
    
function squregion(L::T; layer=0) where {T <: Number}
    
    t = linspace(-L/2, L/2, L+1)
    o = ones(L+1)
    X = [(t * o')[:]'; (o * t')[:]']
    
    X = [X; zeros(size(X, 2))'] |> vecs
    iBC = IBC(Int(L+1); k=layer)
    return X, iBC
        
end
    
function parregion(A::Array{Float64, 2}, L::T; vacancies=0, layer=0, task="2DPoint") where {T <: Number}
    
    # linspace ...
    t = linspace(-L/2, L/2, L+1)
    o = ones(L+1)
    X = A * [(t * o')[:]'; (o * t')[:]']
    
    # vacancies defined here (for single vacancy & crack)
    # vacancies like this [-2, -1, 0, 1, 2, ...]
    if task == "2DPoint"
        iva = Int64((L+2)*L/2+1) + vacancies
    else
        iva = 0
    end
    
    # need rewise to multi-vacancy
    # iva1 = Int64( (L+2)*((L+2-2*layer)/4 + layer) )   # L IS NOT CORRECT! MAN!!
    # iva2 = Int64( (L+1)*((L+2-2*layer)/4 + layer) + (L+2-2*layer)/4*3 + layer )
    # iva3 = Int64( (L+1)*((L+2-2*layer)/4*3 + layer) + (L+2-2*layer)/4 + layer )
    # iva4 = Int64( (L+2)*((L+2-2*layer)/4*3 + layer) ) 
    # iva = [iva1..., iva2..., iva3..., iva4]
    
    iremain = setdiff(1:size(X, 2), iva)
    X = X[:, iremain]
    X = [X; zeros(size(X, 2))'] |> vecs  
    
    # (only for single vacancy/four corner vacancies now)
    iBC = IBC(Int64(L+1); k=layer)
    iBCva = iBC_update!(iBC, iva)
    return X, iBCva
        
end
    
# copy from AC2D codes
function hexregion(L::T; layer=0) where T <: Number
    # lattice directions, auxiliary operators
    a1 = [1, 0]
    Q6 = [cos(pi/3) -sin(pi/3); sin(pi/3) cos(pi/3)]
    a2 = Q6 * a1
    a3 = Q6 * a2

    # no atomistic core first
    K0 = 0
    X = a1 * 0.0
    nX = 2 * K0 + 1
    c = [nX, nX, 1, 1, 1, nX]
    idx = [0 for i=1:L]
    L1 = 2 * K0
    L2 = 0
    for j = 1:Int(L)
        
        L1 += 1; o1 = ones(1, L1)
        L2 += 1; o2 = ones(1, L2)

        X = hcat(X, (X[:, c[1]] + a1) * o2 .+ a3 * collect(0:L2-1)', 
                    (X[:, c[2]] + a2) * o1 .- a1 * collect(0:L1-1)',  
                    (X[:, c[3]] + a3) * o2 .- a2 * collect(0:L2-1)', 
                    (X[:, c[4]] - a1) * o2 .- a3 * collect(0:L2-1)', 
                    (X[:, c[5]] - a2) * o1 .+ a1 * collect(0:L1-1)', 
                    (X[:, c[6]] - a3) * o2 .+ a2 * collect(0:L2-1)')

        c[1] = c[6] + L2 - 1 
        if j == 1
            c[1] = c[1] + 1
        end
        c[2] = c[1] + L2
        c[3] = c[2] + L1
        c[4] = c[3] + L2
        c[5] = c[4] + L2
        c[6] = c[5] + L1
        
        idx[j] = size(X, 2)

    end

    X = [X; zeros(size(X, 2))'] |> vecs
    iBC = collect(idx[Int(L-layer)]+1:idx[end]) 
    
    return X, iBC
    
end

function balregion(L::T; layer=0) where {T <: Number}
    
    A = [1.0 sin(π/6); 0.0 cos(π/6)]
    t = linspace(-L, L, 2L+1)
    o = ones(2L+1)
    X = A*[(t * o')[:]'; (o * t')[:]']
    X = X[:, find(sqrt.(sum(abs2, X, 1)) .< L)]
    iFree = find(sqrt.(sum(abs2, X, 1)) .< L-layer)
    
    X = [X; zeros(size(X, 2))'] |> vecs
    iBC = setdiff(1:length(X), iFree)
    
    return X, iBC
    
end

function IBC(N::Int64; k=0)

    i1 = collect(1:k*N)
    i2 = collect(N^2-k*N+1:N^2)
    i3 = []; i4 = [];
    for i = 1:k
        i3 = vcat(i3, collect(i:N:N^2-N+i))
        i4 = vcat(i4, collect(N-i+1:N:N^2-i+1))
    end
    idx = [i1; i2; i3; i4]
    idx = unique(idx)
    idx = sort(idx)
    
    return idx
    
end  
            
function iBC_update!(iBC, iva)
    
    # now only for sigle vacancy!
    n = length(iBC)
    # println(iBC[Int(n/2)+1:end])
    iBC = vcat(iBC[1:Int(n/2)], iBC[Int(n/2)+1:end] .- 1)
    
    # multi-vacancies!
    # n = length(iBC)
    # I = copy(iBC)
    # only for four vacancies, should develop to any nv...
    # for i = 1:n
    #     if iBC[i] < iva[1]
    #         I[i] = iBC[i]
    #         elseif iva[1] < iBC[i] < iva[2]
    #             I[i] = iBC[i] - 1
    #         elseif iva[2] < iBC[i] < iva[3]
    #             I[i] = iBC[i] - 2
    #         elseif iva[3] < iBC[i] < iva[4]
    #             I[i] = iBC[i] - 3
    #     else
    #         I[i] = iBC[i] - 4
    #     end
    # return I
    # end
    
    # micro-crack!
    # nv = length(iva)
    # n = length(iBC)
    # iBC = vcat(iBC[1:Int(n/2)], iBC[Int(n/2)+1:end] - nv)    
          
end