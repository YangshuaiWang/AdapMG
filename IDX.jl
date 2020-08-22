# Read the coarse mesh from deal ii mesh, construct the geom
# deal ii: bilinear mesh or tri-linear mesh
# Author: Yangshuai Wang, 2019

# TODO: Full the struct...

# include("Tools.jl") I don't want include this, seems silly...

# call JuLIPMaterials for edge dislocation 3D
# include("./JuLIPMaterialsCodes/src/JuLIPMaterials.jl")
# include("./JuLIPMaterialsCodes/src/Si.jl")
# include("./JuLIPMaterialsCodes/src/CauchyBorn_Si.jl")

# for getting basic geometry
include("getX.jl")
include("Tools.jl")

# for read information from deal ii
include("readIO.jl")

# for 2D edge
include("2Dedge_pre.jl")


using JuLIP

# index information including the hanging nodes
struct IDX
    
    idx::Array{Int64, 1}    # find each atom in which element
    iclo::Array{Int64, 1}  # find closest atom
    hanging_index::Array{Any, 1}
    hanging_info::Array{Any, 1}
    hanging_coeff::Array{Any, 1}
    
end

# * Do not need input.
IDX() = IDX(idx, iclo, hanging_index, hanging_info, hanging_coeff) 

# Main function
function updateIDX(xa::Array{Float64, 2}, oldmeshname::Any, meshname::String, filename::String, ii::Any; task = "3DPoint")
# Input: meshname, mesh file name from deal ii, "grid-0.msh"...
#        filename, constraints information from deal ii, "constraints-0.txt"...
#        L, differenet meaning for different task
#        task, 2DPoint, 2DDislocation, 3DPoint and 3DDislocation
# Output: IDX structure...
        
    # read mesh from deal ii
    X1, T1 = read_mesh_msh_dealii_3D(meshname)
    LX1 = maximum(xa) * X1
    if oldmeshname == []
        nX, nT = size(LX1, 2), size(T1, 2)
        h, oT = zeros(nT), zeros(3, nT)
        for j = 1:nT 
            xTj = LX1[:, T1[:, j]]
            h[j] = norm(xTj[:, 1].-xTj[:, 2], Inf)
            oT[:, j] = 1/8 * sum(xTj, 2)
        end
        idxnew = 0 * Array{Int64, 1}(size(xa, 2))
        iclonew = 0 * Array{Int64, 1}(nX)
        for i = 1:nT
            idxnew[find(maximum(abs.(xa .- oT[:, i]), 1)' .<= h[i]/2 + 1e-10)] = i
        end
        for k = 1:nX
            iclonew[k] = indmin(sum(abs2, xa .- LX1[:, k], 1))  # find the closest atom
        end         
    else 
        X0, T0 = read_mesh_msh_dealii_3D(oldmeshname) 
        LX0 = maximum(xa) * X0
        δX = copy(LX1[:, (size(LX0,2)+1):end])
        iclo = 0 * Array{Int64, 1}(size(δX, 2))
        for k = 1:size(δX, 2)
            iclo[k] = indmin(sum(abs2, xa .- δX[:, k], 1))  # find the closest atom
        end
        iclonew = [ii.iclo; iclo]
        idxnew = [ii.idx; 0]  # need improve...
    end

    # read constraints from deal ii 
    hanging_index, hanging_info, hanging_coeff = read_constraints_dealii_3D(filename)

    return IDX(idxnew, iclonew, hanging_index, hanging_info, hanging_coeff)
      
    
end

@inline function myindmin(xa::Array{Float64,2}, X::Array{Float64,2}) 
I = 0 * Array{Int64,1}(size(X, 2))
@simd for k = 1:size(X, 2)
        @inbounds a = sum(abs2, xa .- X[:, k], 1)
        I[k] = indmin(a)
      end 
    return I
end

@inline function get_initial_idx(h::Float64, oT::Array{Float64,2})   
    ixa = get_div_index(xa, h) 
    ioT = get_div_index(oT, h) 
    AA = set_index!(ioT)
    idx = get_index(ixa, AA)
    Idx = update_index(idx, oT)
    return Idx
end

@inline get_div_index(x::Union{Array{Float64,2}, Array{Int64,1}}, h::Float64) = map(Int64, div.(x, h+1e-6)+1.0)

@inline function set_index!(ioT::Array{Int64,2})
    noT = Int64(maximum(ioT))
    AA = 0 * Array{Int64,3}(noT, noT, noT)
    @simd for i = 1:size(ioT, 2)
        AA[ioT[1, i], ioT[2, i], ioT[3, i]] = i
    end
    return AA
end

@inline function get_index(ixa::Array{Int64,2}, AA::Array{Int64,3})
    nxa = size(xa, 2)
    idx = 0 * Array{Int64,1}(nxa)
    @simd for j = 1:nxa
        idx[j] = AA[ixa[1, j], ixa[2, j], ixa[3, j]]
    end
    return idx
end

@inline function update_index(idx::Array{Int64,1}, oT::Array{Float64,2})
    Idx = Array{Int64,1}[]
    @simd for z = 1:length(oT)
        iz = findin(idx, z)
        push!(Idx, iz)
    end
    return Idx
end
    
function update_Idx(Idx::Array{Array{Int64,1},1}, η::Array{Float64,1}, ngT::Array{Int64,2}, gT::Array{Int64,2}, ngX::Array{Float64,2}, gX::Array{Float64,2}, oT::Array{Float64,2})
    n_ngT, n_gT = size(ngT, 2), size(gT, 2)
    k = (n_ngT - n_gT) ÷ 7
    ITref = sortperm(η, rev=true)[1:k]  # the index that to be refined 
    InonTref = setdiff(1:n_gT, ITref)
    nonTref = gT[:, InonTref]
    Tref = gT[:, ITref] # get_T of to be refined
    # find interiori points for each Tref
    oo = zeros(3, k)
    Iint = 0 * Array{Int64,1}(k) 
    @simd for ii = 1:k
        oo[:, ii] = 1/8 * sum(gX[:, Tref[:, ii]], 2)
        Iint[ii] = find(maximum(abs.(ngX .- oo[:, ii]), 1)' .<= 1e-3)[1]
    end
    Idxnew = initial_empty(n_ngT)
    in_ig = setdiff_T(n_gT-k, ngT, nonTref)
    Idxnew[in_ig] = Idx[InonTref]
    @simd for j = 1:k
        index_in_Tref = Idx[ITref[j]] 
        xa_in_I = xa[:, index_in_Tref] 
        oT_in_Tref = 1/size(xa_in_I, 2)*sum(xa_in_I, 2)#oT[:, ITref[j]]
        delta_xa_in_I = map(Int64, sign.(xa_in_I .- oT_in_Tref)[:])
        idx_matrix = map_sign_to_idx!(delta_xa_in_I) 
        
        # silly implementation
        Isilly = initial_empty(8)
        for m = 1:length(index_in_Tref)
            push!(Isilly[index_tensor[idx_matrix[1,m], idx_matrix[2,m], idx_matrix[3,m]]], m)
        end
        for isi = 1:8
            ooo = 1/size(xa[:, index_in_Tref[Isilly[isi]]], 2)*sum(xa[:, index_in_Tref[Isilly[isi]]], 2)
            oook = indmin(sum(abs2, oT .- ooo, 1))
            Idxnew[oook] = index_in_Tref[Isilly[isi]]
        end
                    
        
# fast implementation -- need debug !        
#         index_in_newgeom = findin(ngT[:], Iint[j]) 
#         iii = get_div_index(index_in_newgeom, 8.0)
#         sorted_refined_element = sortcols((hcat(mod.(index_in_newgeom, 8).+1, iii)'))[2, :]  
#         @simd for m = 1:length(index_in_Tref)
#             iiii = sorted_refined_element[index_tensor[idx_matrix[1,m], idx_matrix[2,m], idx_matrix[3,m]]]
#             push!(Idxnew[iiii], index_in_Tref[m])
#         end
    end
    return Idxnew
end

@inline function initial_empty(n_ngT::Int64)
    Idxnew = Array{Int64,1}[]
    @simd for i = 1:n_ngT
        push!(Idxnew, Array{Int64, 1}[])
    end
    return Idxnew
end

@inline function setdiff_T(nn::Int64, ngT::Array{Int64,2}, nonTref::Array{Int64,2})
    in_ig = 0 * Array{Int64,1}(nn)
    @simd for g = 1:nn
        in_ig[g] = findin(ngT[1, :], nonTref[1, g])[1]
    end
    return in_ig
end

function compute_iδ(ngX::Array{Float64,2}, ngT::Array{Int64,2}, gX::Array{Float64,2}, Idxnew::Array{Array{Int64,1},1})
    δX = ngX[:, (size(gX, 2)+1):end]
    δidx = 0 * Array{Int64, 1}(size(δX, 2))
    @simd for d = 1:size(δX, 2)
        iδx = size(gX, 2) + d
        iδx_neigh = get_div_index(findin(ngT[:], iδx), 8.0)
        II = Idxnew[iδx_neigh[1]]
        @simd for i = 2:length(iδx_neigh)
            @inbounds II = vcat(II, Idxnew[iδx_neigh[i]])
        end
        δx_neigh_xa = xa[:, II]
        δidx[d] = II[indmin(sum(abs2, δx_neigh_xa .- δX[:,d], 1))] # II[...]
    end
    return δidx
end

# index_tensor = 0 * Array{Int64, 3}(3, 3, 3)
# index_tensor[1, 1, 1] = 7
# index_tensor[2:3, 1, 1] = 8
# index_tensor[2:3, 1, 2:3] = 5
# index_tensor[1, 1, 2:3] = 6
# index_tensor[1, 2:3, 1] = 3
# index_tensor[2:3, 2:3, 1] = 4
# index_tensor[2:3, 2:3, 2:3] = 1
# index_tensor[1, 2:3, 2:3] = 2;

index_tensor = 0 * Array{Int64, 3}(2, 2, 2)
index_tensor[1, 1, 1] = 7
index_tensor[2, 1, 1] = 8
index_tensor[2, 1, 2] = 5
index_tensor[1, 1, 2] = 6
index_tensor[1, 2, 1] = 3
index_tensor[2, 2, 1] = 4
index_tensor[2, 2, 2] = 1
index_tensor[1, 2, 2] = 2;

function map_sign_to_idx!(S::Array{Int64,1})
    for i = 1:length(S)
        if S[i] == -1
            S[i] = 1
        else
            S[i] = 2
        end
    end
    return reshape(S, 3, length(S)÷3)
end   

