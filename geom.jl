# Read the coarse mesh from deal ii mesh, construct the geom
# deal ii: bilinear mesh or tri-linear mesh
# Author: Yangshuai Wang, 2019

# TODO: Full the struct...

# include("Tools.jl") I don't want include this, seems silly...

# call JuLIPMaterials for edge dislocation 3D
include("./JuLIPMaterialsCodes/src/JuLIPMaterials.jl")
include("./JuLIPMaterialsCodes/src/Si.jl")
include("./JuLIPMaterialsCodes/src/CauchyBorn_Si.jl")

# for getting basic geometry
include("getX.jl")
include("Tools.jl")

# for read information from deal ii
include("readIO.jl")

# for 2D edge
include("2Dedge_pre.jl")


using JuLIP, PyPlot

# at should not in this function
struct dealii_geom
    
    s::String  # task
    L::Int64  # model params
    i::Int64  # step info
    at::JuLIP.Atoms{Float64, Int64}  # atomistic scaled Atoms
    X::Array{Float64, 2}  # mesh position
    T::Array{Int64, 2}  # mesh index
    ifree::Array{Int64, 1}  # think about this, whether ifree or ibc ???
#     idx::Array{Int64, 1}    # find each atom in which element
#     iclo::Array{Int64, 1}  # find closest atom
    A::Array{Float64, 2}   # A indicates the dimension
    oT::Array{Float64, 2}  # coodinate of mid-point of element
    h::Array{Float64, 1}  # mesh size
    dh::Array{Float64, 1}  # distance from defect
    hmin::Float64  # minimum mesh 
    hanging_index::Array{Any, 1}
    hanging_info::Array{Any, 1}
    hanging_coeff::Array{Any, 1}
    
end

# * Do not need input.
dealii_geom() = dealii_geom(s, L, i, at, X, T, ifree, A, oT, h, dh, hmin, hanging_index, hanging_info, hanging_coeff) 

# Main function
function readmsh(at::JuLIP.Atoms{Float64, Int64}, meshname::String, filename::String, L::Int64, i::Int64; task = "3DPoint")
# Input: meshname, mesh file name from deal ii, "grid-0.msh"...
#        filename, constraints information from deal ii, "constraints-0.txt"
#        L, differenet meaning for different task
#        task, 2DPoint, 2DDislocation, 3DPoint and 3DDislocation
# Output: geom structure...
    
    if task == "2DPoint"
    
        # read mesh from deal ii
        Xo, To = read_mesh_msh_dealii(meshname)

        # scaled square mesh
        X = L * Xo .- L / 2 

        # deformation A
        A = [1.0 sin(π/6); 0.0 cos(π/6)]
        lay = 3; # outside layer
        va = 0; # only for single vacancy + multi-vacan
        # va = [-3, -2, -1, 0, 1, 2, 3] # micro-crack

        # form the atomistic configuration with vacancy
        x, ibc = getX(A, L+2*lay, va, :parallel, layer=lay, task=task)

        # construct at
        at = Atoms(:X, x)
        set_calculator!(at, lennardjones()*SplineCutoff(1.8, 2.7))
        set_constraint!(at, FixedCell2D(at, clamp=ibc));
        
        xa = mat(x)[1:2, :]

        # get DoF
        ifree = setdiff(1:size(xa, 2), ibc)
        xrefa = xa[:, ifree]
        xref = A\xrefa

        # obtain the index of each element
        idx = Array{Int64, 1}(zeros(length(ifree))) # exists no overlap!!
        for i = size(To, 2):-1:1
            _, ii = idxElement2Atom(X, To, xref, i)
            idx[ii] = i
        end
        
        # read constraints from deal ii
        hanging_index, hanging_info, hanging_coeff = read_constraints_dealii(filename)
    
        return dealii_geom(L, i, X, To, ifree, idx, A, at, hanging_index, hanging_info, hanging_coeff)
        
    end
    
    if task == "2DDislocation"
        
        # read mesh from deal ii
        Xo, To = read_mesh_msh_dealii(meshname)

        # deformation A
        A = [1.0 sin(π/6); 0.0 cos(π/6)]
        lay = 3; va = 0; 
        
        # form the atomistic configuration 
        x, ibc = getX(A, L+2*lay, va, :parallel, layer=lay, task=task)
        at = Atoms(:X, x)
        
        # with dislocation predictor...
        X = positions(at)
        # find center-atom
        F = [1.0 0.0 0.0; 0.0 √3 0.0; 0.0 0.0 1.0]
        x0 = JVec([0.5 * diag(F)[1:2]; 0.0])
        _ , I0 = findmin([norm(x - x0) for x in X])
        # dislocation core
        tcore = JVec([0.5, √3/6, 0.0])
        xc = X[I0] + tcore
        # shift configuration to move core to 0
        X = [x - xc for x in positions(at)]
        set_positions!(at, X)
        # remove the center-atom
        deleteat!(at, I0)
        # apply dislocation FF predictor
        edge_predictor!(at; b=1.0, xicorr=true, ν=0.25)
        X = [x + tcore for x in positions(at)]
        set_positions!(at, X)
        
        # scaled square mesh
        xa = A \ mat(at.X)[1:2, :]
        Lmax = maximum(abs.(xa))
        X = 2 * Lmax * Xo - Lmax
        
        Rqm = Lmax - 5
        r = norm.(positions(at))
        Ifree = find(r .<= Rqm)
        set_calculator!(at, lennardjones()*SplineCutoff(1.8, 2.7))
        set_constraint!(at, JuLIP.Constraints.FixedCell2D(at; free = Ifree))
        
        
        nT = size(To, 2)
        oT = zeros(2, nT)
        h = zeros(nT)
        for j = 1:nT 
            xTj = X[:, To[:, j]]
            h[j] = norm(xTj[:, 1].-xTj[:, 2], Inf)
            oT[:, j] = 1/4 * sum(xTj, 2)
        end        
        tic()
        idx = 0 * Array{Int64, 1}(size(xa, 2))
        for i = 1:size(To, 2)
            idx[find(maximum(abs.(xa .- oT[:, i]), 1)' .<= h[i]/2 + 1e-10)] = i
        end
        toc()
           
        # read constraints from deal ii
        hanging_index, hanging_info, hanging_coeff = read_constraints_dealii(filename)
    
        return dealii_geom(X, To, Ifree, idx, A, hanging_index, hanging_info, hanging_coeff)
        
    end
        
    if task == "3DPoint"
        
        # read mesh from deal ii
        Xo, To = read_mesh_msh_dealii_3D(meshname)
        
        # scaled cubic mesh
        xa = mat(at.X)
        Lmax = maximum(xa)
        X = Lmax * Xo
        
        # boundary part, define casually, do not used.
        ifree = [0]
        A = [1.0 0.0 0.0; 0.0 √3 0.0; 0.0 0.0 1.0]
        
        od = get_data(at, "defpos")
        nX, nT = size(Xo, 2), size(To, 2)
        h, dh, oT = zeros(nT), zeros(nT), zeros(3, nT)
        @simd for j = 1:nT 
            xTj = X[:, To[:, j]]
            h[j] = norm(xTj[:, 1].-xTj[:, 2], Inf)
            oT[:, j] = 1/8 * sum(xTj, 2)
            dh[j] = minimum(sqrt.(sum(abs2, oT[:,j] .- od, 1)))
        end
        hmin = minimum(h)

        # read constraints from deal ii 
        hanging_index, hanging_info, hanging_coeff = read_constraints_dealii_3D(filename)

        return dealii_geom(task, L, i, at, X, To, ifree, A, oT, h, dh, hmin, hanging_index, hanging_info, hanging_coeff)
        
    end
    
    if task == "3DDislocation"
        
        # read mesh from deal ii
        Xo, To = read_mesh_msh_dealii_3D(meshname)

        # construct atoms
        at, _ = Si.edge110(:Si, L)
        set_constraint!(at, FixedCell(at))
        
        # scaled cubic mesh
        xa = mat(at.X)
        Lmax = maximum(abs.(xa))
        Lmin = minimum(xa)
        LL = Lmax - Lmin
        X = LL * Xo + Lmin
        ########################### better to put the at in the middle...or change the cluster coefficient...
        
        # boundary part
        ifree = [0]
        A = [1.0 0.0 0.0; 0.0 √3 0.0; 0.0 0.0 1.0]
        
        nT = size(To, 2)
        oT = zeros(3, nT)
        h = zeros(nT)
        for j = 1:nT 
            xTj = X[:, To[:, j]]
            h[j] = norm(xTj[:, 1].-xTj[:, 2], Inf)
            oT[:, j] = 1/8 * sum(xTj, 2)
        end
        
        tic()
        idx = 0 * Array{Int64, 1}(size(xa, 2))
        for i = 1:size(To, 2)
            idx[find(maximum(abs.(xa .- oT[:, i]), 1)' .<= h[i]/2 + 1e-10)] = i
        end
        toc()
        # !!!!!! NEED IT EFFECIENT !!!!!!!!!!!
        
        # read constraints from deal ii 
        hanging_index, hanging_info, hanging_coeff = read_constraints_dealii_3D(filename)
        
        return dealii_geom(X, To, ifree, idx, A, hanging_index, hanging_info, hanging_coeff)
    end
    
end


@inline findinT(xa::Array{Float64,2}, oT::Array{Float64,2}, h::Array{Float64,1}) = find(maximum(abs.(xa .- oT[:, i]), 1)' .<= h[i]/2 + 1e-10)






