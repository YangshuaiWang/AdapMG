using JuLIP, Optim, LineSearches, HDF5, JLD

# need the basis interpolation
include("Tools.jl")

# deal with hanging nodes
# extend to hanging nodes version (newest version!)
# to more general case, not only 0.5...
function extendU(geom, Uold)
    Idof = setdiff(1:N, geom.hanging_index)
    U = zeros(size(geom.A, 1), size(geom.X, 2)) 
    U[:, Idof] = copy(Uold)
    @simd for i = 1:length(geom.hanging_index)
        # U[:, Ihang[i]] = 0.5*(U[:, geom.hanging_info[i][1]]+U[:, geom.hanging_info[i][2]])
        @inbounds U[:, geom.hanging_index[i]] = geom.hanging_coeff[i]' * U[:, geom.hanging_info[i]]
    end   
    return U
end

# need in other .jl file...
"""
Write a txt file to be read by deal-ii ifstream
! The input u must be a vector so far!
"""
function write_estimator(filename, u)
	dl = "\t"
	nu = length(u)
	nloop = Int64(floor(nu/11))
	open(filename, "w") do f
		#println(f, nu, dl)
		for i = 1:nloop
			n = (i-1)*11
			println(f, dl, u[n+1], dl, u[n+2], dl, u[n+3], dl, u[n+4], dl, u[n+5], dl, u[n+6], dl, u[n+7], dl, u[n+8], dl, u[n+9], dl, u[n+10], dl, u[n+11])
		end
		str = "\t"
		for i = (11*nloop+1):nu
			str = str*string(u[i])*dl
		end
		println(f, str)
	end
end

function write_iter_number(filename, num)
	dl = "\t"
	open(filename, "w") do f
		println(f, num)
	end
end

function write_refine_coeff(filename, cc)
	dl = "\t"
	open(filename, "w") do f
		println(f, cc)
	end
end

# use the interpolation to get the initial...
# need old layer geom
function get_initial(geom, oldgeom, U)
    
    xc = oldgeom.X
    Tc = oldgeom.T
    N = size(geom.X, 2)
    Ihang = geom.hanging_index
    Nhang = length(Ihang)
    Idof = setdiff(1:N, Ihang)
    xf = geom.X[:, Idof]
    # can be written as a interpolation function actually
    u = zeros(2, N-Nhang)
    for i = 1:size(xf, 2)
        # need revise
        # check in which old mesh element
        for j = 1:size(Tc, 2)
            xT = xc[:, Tc[:, j]]
            if (xT[1, 1] <= xf[1, i] <= xT[1, 3]) & (xT[2, 1] <= xf[2, i] <= xT[2, 3])
                xsq = para2square(xT, xf[:, i])
                uT = U[:, Tc[:, j]]
                @inbounds w, uu = biInterp(xsq, uT)
                u[:, i] = copy(uu)
            end
        end
    end
    return u
end

function get_index_n(Idx, iA_in_T)
    i = Idx[iA_in_T[1]]
    for j = 2:length(iA_in_T)
        i = vcat(i, Idx[iA_in_T[j]]...)
    end
    return i
end

function gradientQCE(atAbuf, geom, x)
    
    dE = zeros(3, size(x, 2))
    for k in setdiff(1:nT, iTref)
        X = geom.X[:, geom.T[:, k]]'
        u = x[:, length(idxAB).+geom.T[:, k]]
        wX_T = geom.h[1] * wX .+ X[1,:]
        for i = 1:27
            Du = compute_Du(X, geom.h[1], wX_T[1,i], wX_T[2,i], wX_T[3,i], u) # prob ????
            dE[:, length(iA)+1:end] += len[k] * w[i] * CauchyBorn.grad(W, eye(3)+Du)
        end
    end
    
    # recompute the buf displacement
    δu = zeros(3, length(idxAB))
    # iB = setdiff(idxAB, iA)
    iB_in_T = myindmin(geom.oT, xa[:,iB])
    # iI = findin(idxAB, iB)
    for iiB = 1:length(iB_in_idxAB)
        δu[:, iB_in_idxAB[iiB]] = compute_u(xa[:,iB[iiB]], geom.X[:,geom.T[:,iB_in_T[iiB]]], x[:,length(iA).+geom.T[:,iB_in_T[iiB]]], geom.h[1])
    end
    
    # EA   
    δu[:, iB_in_idxAB] = copy(ubuf)
    δu[:, iA_in_idxAB] = x[:, length(iA)]
    set_positions!(atAbuf, mat(atAbuf.X) + δu) # where X is from interpolated
    dE[:, 1:length(iA)] = forces(atAbuf)[:, iA]
    
    return dE
end


function energyQCE(atAbuf, geom, x)
    # the input x contains two parts: the first is idxAB, the second is geom.X 
    # maybe the displacement u is better
    
    iA_in_idxAB = findin(idxAB, iA)
    iB_in_idxAB = setdiff(1:length(idxAB), iA_in_idxAB)
    iB = idxAB[iB_in_idxAB]
    
    # EC
    EC = 0
    for k in setdiff(1:nT, iTref)
        X = geom.X[:, geom.T[:, k]]'
        u = x[:, length(idxAB).+geom.T[:, k]]
        wX_T = geom.h[1] * wX .+ X[1,:]
        for i = 1:27
            Du = compute_Du(X, geom.h[1], wX_T[1,i], wX_T[2,i], wX_T[3,i], u) # prob ????
            EC += len[k] * w[i] * W(eye(3)+Du) 
        end
    end
    
    # recompute the buf displacement
    δu = zeros(3, length(idxAB))
    # iB = setdiff(idxAB, iA)
    iB_in_T = myindmin(geom.oT, xa[:,iB])
    # iI = findin(idxAB, iB)
    for iiB = 1:length(iB_in_idxAB)
        δu[:, iB_in_idxAB[iiB]] = compute_u(xa[:,iB[iiB]], geom.X[:,geom.T[:,iB_in_T[iiB]]], x[:,length(iA).+geom.T[:,iB_in_T[iiB]]], geom.h[1])
    end
    
    # EA   
    δu[:, iB_in_idxAB] = copy(ubuf)
    δu[:, iA_in_idxAB] = x[:, length(iA)]
    set_positions!(atAbuf, mat(atAbuf.X) + δu) # where X is from interpolated
    EA = 0
    for isite in iA_in_idxAB
        EA += site_energy(atAbuf.calc, atAbuf, isite)
    end
    
    E = EA + EC
    return E
end

# maybe try static array
@inline function mydot(x::Array{Float64,2})
    S = x[1, :]
    @simd for j = 2:size(x,1)
        @inbounds S .*= x[j, :]
    end
    return S
end

# fast implementation in each element
function compute_u(x, X, U, h)
    # X should be 3*8
    # U should be 3*8
    # x should be 3*27 (for integral) or 3*N (for interpolation)
    ϕ = zeros(size(x,2), 8)
    for i = 1:8
        unϕ = (1 .- abs.(x.-X[:,i])/h)
        ϕ[:, i] = mydot(unϕ) 
    end
    u = U * ϕ'
    return u
end
# compute_Du shoule at least 3, \partial_x, \partial_y, \partial_z


    
