function myplotgeom(geom)
    Ii = [1 2 3 4 1 2 3 4 5 6 7 8; 2 3 4 1 5 6 7 8 6 7 8 5]
    for iT = 1:size(geom.T, 2)
        T = geom.T[:, iT]
        for iI = 1:12
            i = Ii[:, iI]
            xx = geom.X[:, T[i]]
            plot3D(xx[1,:], xx[2,:], xx[3,:], "bo-", linewidth=0.7, markersize=4)
        end
    end     
    xticks([])
    yticks([])
    zticks([])
    pause(0.5)
end

function myplotA(x::Array{Float64,2})
    plot3D(x[1,:], x[2,:], x[3,:], "ro", markersize=1)
    xticks([])
    yticks([])
    zticks([])
    pause(0.5)
end

function myplotQCE(geom, x::Array{Float64,2})
    Ii = [1 2 3 4 1 2 3 4 5 6 7 8; 2 3 4 1 5 6 7 8 6 7 8 5]
    for iT = 1:size(geom.T, 2)
        T = geom.T[:, iT]
        for iI = 1:12
            i = Ii[:, iI]
            xx = geom.X[:, T[i]]
            plot3D(xx[1,:], xx[2,:], xx[3,:], "bo-", linewidth=0.7, markersize=4)
        end
    end    
    plot3D(x[1,:], x[2,:], x[3,:], "ro", markersize=1)
    xticks([])
    yticks([])
    zticks([])
    pause(0.5)   
end