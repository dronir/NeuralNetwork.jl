module NeuralNetwork


sigmoid(x) = 1.0 ./ (1.0 + exp(-x))
sigmoidprime(x) = sigmoid(x) * (1.0 - sigmoid(x))

immutable Network
	Nlayers::Int64
	sizes::Array{Int64,1}
	biases::Array{Array{Float64,2},1}
	weights::Array{Array{Float64,2},1}
end

function Network(sizes::Array{Int64,1})
	N = length(sizes)
	B = [randn(y,1) for y = sizes[2:end]]
	W = [randn(y,x) for (x,y) = zip(sizes[1:end-1], sizes[2:end])]
	Network(N, sizes, B, W)
end

# Output of the whole network for input a
function feedforward(N::Network, a::Array{Float64,2})
	for (b,w) = zip(N.biases, N.weights)
		a = sigmoid(w*a + b)
	end
	return a
end



end #module
