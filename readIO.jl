#####################################################################################################
# should be put into "read.jl"

# read mesh information in 2D
function read_mesh_msh_dealii(fn::AbstractString)
open(fn, "r") do io
    read_mesh_msh_dealii(io)
    end
end

function read_mesh_msh_dealii(io)
    dim = 2
    thisLine = io |> readline |> strip
    thisLine = io |> readline |> strip
    NV = parse(Int, thisLine)
	X = zeros(dim+2, NV)
    for i in 1:NV
        thisLine = io |> readline |>  strip
        d = readdlm(IOBuffer(thisLine), Float64)
		X[:,i] = d
    end
    
    thisLine = io |> readline |> strip
    thisLine = io |> readline |> strip
    thisLine = io |> readline |> strip
    NV = parse(Int, thisLine)
    T = Array{Int64, 2}(zeros(dim+7, NV))
    for i in 1:NV
        thisLine = io |> readline |>  strip
        d = readdlm(IOBuffer(thisLine), Int64)
		T[:,i] = d
    end
    return X[2:3, :], T[6:end, :]
end


# read mesh information in 3D
function read_mesh_msh_dealii_3D(fn::AbstractString)
open(fn, "r") do io
    read_mesh_msh_dealii_3D(io)
    end
end

function read_mesh_msh_dealii_3D(io)
    dim = 3
    thisLine = io |> readline |> strip
    thisLine = io |> readline |> strip
    NV = parse(Int, thisLine)
	X = zeros(dim+1, NV)
    for i in 1:NV
        thisLine = io |> readline |>  strip
        d = readdlm(IOBuffer(thisLine), Float64)
		X[:,i] = d
    end
    
    thisLine = io |> readline |> strip
    thisLine = io |> readline |> strip
    thisLine = io |> readline |> strip
    NV = parse(Int, thisLine)
    T = Array{Int64, 2}(zeros(dim+10, NV))
    for i in 1:NV
        thisLine = io |> readline |>  strip
        d = readdlm(IOBuffer(thisLine), Int64)
		T[:,i] = d
    end
    return X[2:4, :], T[6:end, :]
end


# read haning nodes function in 2D
function read_constraints_dealii(fn::AbstractString)
    open(fn, "r") do f
        Con = []
        line = 1
        while !eof(f)
            thisLine = f |> readline |>  strip
            d = readdlm(IOBuffer(thisLine))
            push!(Con, d)     
            line += 1
        end

        for i = 1:line-1
            Con[i][2] = parse(Int64, split(Con[i][2], ":")[1])
        end
    
        c = zeros(2, line-1)
        for i = 1:2
            for j = 1:line-1
                c[i, j] = Con[j][i]
            end
        end
        
        hanging_nodes_index = Array{Int64}(c[1, :])[1:2:end]
        hanging_nodes_info = [Array{Int64}(c[2, :][2*i-1:2*i]) for i = 1:length(hanging_nodes_index)]
        hanging_nodes_coeff = [[0.5, 0.5] for i = 1:length(hanging_nodes_index)]
        
        return hanging_nodes_index, hanging_nodes_info, hanging_nodes_coeff
    end
end


# read haning nodes function in 3D
function read_constraints_dealii_3D(fn::AbstractString)
    open(fn, "r") do f
        Con = []
        line = 1
        while !eof(f)
            thisLine = f |> readline |>  strip
            d = readdlm(IOBuffer(thisLine)) 
            push!(Con, d)     
            line += 1
        end
        
        for i = 1:line-1
            Con[i][2] = parse(Int64, split(Con[i][2], ":")[1])           
        end
          
        c = Array{Int64, 2}(zeros(2, line-1))
        d = Array{Float64, 2}(zeros(2, line-1))
        for i = 1:2
            for j = 1:line-1
                c[i, j] = Con[j][i]
            end
        end
        
        d[1, :] = copy(c[1, :])
        for j = 1:line-1
            d[2, j] = Con[j][3]
        end
               
        hanging_nodes_index = unique(c[1, :])
        hanging_nodes_info = []
        hanging_nodes_coeff = []
        for i =1:length(hanging_nodes_index)
            push!(hanging_nodes_info, c[2, findin(c[1, :], hanging_nodes_index[i])])  
            push!(hanging_nodes_coeff, d[2, findin(d[1, :], hanging_nodes_index[i])])
        end
        
        return hanging_nodes_index, hanging_nodes_info, hanging_nodes_coeff
    end
end

#####################################################################################################