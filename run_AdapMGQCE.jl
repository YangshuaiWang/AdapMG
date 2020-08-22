
# include the function
using JuLIP, HDF5, JLD, PyPlot, Optim, LineSearches
include("./Juliacodes/subfunction.jl")


################################################################################################
# 1. compute the atomistic solution and construct the initial mesh
# FCC-Al, L*L*L cube
L = 16
println("The atomistic configuration (Cu) is a ", L, "*", L, "*", L, " cell.")
at = construct_a_sing(:Cu, L)
# record the reference position
xold = copy(mat(at.X))
xa = copy(xold)

# plot the atomistic configuration that should be solved
solve_refa(L, at, xold)
figure(0)
myplotA(mat(at.X))

# construct the initial geometry
println("Construct the initial mesh !")
geom, idx, Idx, initial = construct_ini_geom(L, at, xa)
inigeom = deepcopy(geom)
iniIdx = copy(Idx)
# the initial mesh is in Figure 0
myplotgeom(geom)
################################################################################################


################################################################################################
# 2. solve the QCE problem
# NEED UPDATE！
# begin with the geom generated above
oldgeom = geom
iA_in_T = []
c = 0.6
println("QCE description to solve !")
nX = size(geom.X, 2)
while true # or nX be the limit # geom.hmin >= 4.0 can not be this

    println("Step ", i, " Begin !")

    # pre_solve
    # println("The number of the atomistic region: ", length(iA))
    # check_element_in_C -> Aregion (or all_element_resetgeom)
    figure(i)
    iA_in_T = find(geom.h .< htol+1) # find(geom.dh .< 4+12*(i-4)) # find(geom.h .< htol) ??? 
    iA = get_index_n(Idx, iA_in_T)
    myplotQCE(geom, xa[:, iA])
    idxAB, iB = atAbuf_info(at, iA)
    atAbuf = construct_atAbuf(idxAB)

    # solve
    u = solve_QCE(atAbuf, geom, iA, idx, initial, optimiser, relxnit)
    
    println("Step ", i, " Done !")

end
################################################################################################


