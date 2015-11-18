module NeuralNetwork


sigmoid(x) = one(x) / (one(x) + exp(-x))

immutable Network
	Nlayers::Int64
	sizes::Array{Int64,1}
	biases::Array{Array{Float64,1},1}
	weights::Array{Array{Float64,2},1}
end

function Network(sizes::Array{Int64,1})
	N = length(sizes)
	B = [randn(y) for y = sizes[2:end]]
	W = [randn(x,y) for (x,y) = zip(sizes[1:end-1], sizes[2:end])]
	Network(N, sizes, B, W)
end



end #module
