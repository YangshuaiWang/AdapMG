# include the function
using JuLIP, HDF5, JLD, PyPlot, Optim, LineSearches
include("myplot.jl")
include("geom.jl")
include("solver.jl")
include("IDX.jl")

function construct_a_sing(symbol, L)
    at = bulk(symbol, cubic=true) * L
    xdef = [1/2 1/2 1/2]' * at.cell[1]
    idef = indmin(sum(abs2, mat(at.X) .- xdef, 1))
    
    at = bulk(symbol, cubic=true) * L
    deleteat!(at, idef)
    set_data!(at, "defpos", xdef)
    # EAM potential, third nearest neighbour interaction
    pth = "/Users/yswang/.julia/v0.6/JuLIP/data/"
    calc = JuLIP.Potentials.FinnisSinclair(pth*"pcu.plt", pth*"fcu.plt")
    set_calculator!(at, calc)
    set_constraint!(at, FixedCell(at)) # periodic boundary condition
    return at
end

function solve_refa(L, at, xold)
    # calculate the reference solution to compare
    if !isfile(string("3DAlPoint", L, ".jld"))
        tic()
        # relax the criteria to 1e-4
        minimise!(at, verbose = 2, gtol = 1e-4, precond = :id)
        println("  Reference time: ")
        ta = toc()
        xnew = copy(mat(at.X))
        u = xnew .- xold
        save(string("Time3DAlPoint", L, ".jld"), "ta", ta)
        save(string("3DAlPoint", L, ".jld"), "u", u)
    end
    # ua = load(string("3DAlPoint", L, ".jld"), "u") # no need
end

function construct_ini_geom(L, at, xa)
    initial_grid = "grid-0.msh"
    initial_cons = "constraints-0.txt"
    geom = readmsh(at, initial_grid, initial_cons, L, 0)
    initial = zeros(2, size(geom.X, 2) - length(geom.hanging_index))
    # allocate each atoms to each element
    Idx = get_initial_idx(geom.h[1], geom.oT)
    idx = myindmin(xa, geom.X)
    return geom, idx, Idx, initial
end

function write_all(η, i, c)
    estname = string("estimator-", i, ".txt")
    write_estimator(estname, η)
    # write the iteration number
    itername = string("iteration.txt")
    write_iter_number(itername, i-1)
    # write the refinement coefficient
    coefname = string("coeff.txt")
    write_refine_coeff(coefname, c)
end

function atAbuf_info(at, iA)
    nlist = neighbourlist(at)
    idxAB = nlist.j[findin(nlist.i, iA)]
    idxAB = unique(idxAB)
    iA_in_idxAB = findin(idxAB, iA)
    iB_in_idxAB = setdiff(1:length(idxAB), iA_in_idxAB)    # iB = setdiff(idxAB, iA)
    iB = idxAB[iB_in_idxAB]
    return idxAB, iB
end

function construct_atAbuf(idxAB)
    # construct the atAbuf to calculate the A energy
    XvecsA = xa[:, idxAB] |> vecs
    atAbuf = Atoms(:Al, XvecsA)
    set_pbc!(atAbuf, false) # not pbc!s
    set_cell!(atAbuf, 5*copy(atAbuf.cell))
    # EAM potential, third nearest neighbour interaction
    pth = "/Users/yswang/.julia/v0.6/JuLIP/data/"
    calc = JuLIP.Potentials.FinnisSinclair(pth*"pcu.plt", pth*"fcu.plt")
    set_calculator!(atAbuf, calc)
    set_constraint!(atAbuf, FixedCell(atAbuf))
    return atAbuf
end

function solve_QCE(atAbuf, geom, iA, idx, initial, optimiser, relxnit)    
    # construct the objective function (E & G) for QCE !
    obj_fqce = x -> energyQCE(atAbuf, geom, x)
    obj_g!qce = (g, x) -> copy!(g, gradientQCE(atAbuf, geom, x))
    
    # call Optim package to solve
    u = optimize(obj_fqce, obj_g!qce, initial, optimiser，Optim.Options(f_tol = 1e-32, g_tol = 1e-4,
                                        store_trace = false,
                                        extended_trace = false,
                                        callback = nothing,
                                        show_trace = true))
    # save important informatins in one loop
    filename = string("step_", i, ".jld")
    save(filename, "u", u) 
    return u
end  

function update_info_QCE(geom, i, L, Idx, η, idx)
    oldgeom = geom
    newgridname = string("grid-", i, ".msh")
    newconstraintname = string("constraints-", i, ".txt")
    geom = readmsh(at, newgridname, newconstraintname, L, i)
    Idx = update_Idx(Idx, η, geom.T, oldgeom.T, geom.X, oldgeom.X, geom.oT)
    δidx = compute_iδ(geom.X, geom.T, oldgeom.X, Idx)  # check !
    idx = [idx; δidx]
    i = i + 1
    return i, idx, Idx, geom 
end

function solve_a(at, initial, optimiser)
    obj_f = x -> energy(at, x)
    obj_g! = (g, x) -> copy!(g, gradient(at, x))
    results = Optim.optimize(obj_f, obj_g!, initial, optimiser,
                            Optim.Options( f_tol = 1e-32, g_tol = 1e-4,
                                        store_trace = false,
                                        extended_trace = false,
                                        callback = nothing,
                                        show_trace = true ))
end