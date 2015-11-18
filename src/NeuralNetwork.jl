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
function feedforward(Net::Network, a::Array{Float64,2})
	for (b,w) = zip(Net.biases, Net.weights)
		a = sigmoid(w*a + b)
	end
	return a
end


function SGD(Net::Network, training_data, 
			 epochs::Integer, batch_size::Integer, eta::Real)
	N = length(training_data)
	for j = 1:epochs
		#shuffle!(training_data)
		batches = [
			training_data[k:k+batch_size-1] for k = 1:batch_size:N-batch_size
		]
		for batch in batches
			train!(Net, batch, eta)
		end
	end
end

function train!(Net::Network, batch, eta::Real)
	M = length(batch)
	nabla_b = [zeros(size(b)) for b in Net.biases]
	nabla_w = [zeros(size(w)) for w in Net.weights]
	println("batch = ", typeof(batch))
	for (x,y) = batch
		delta_nabla_b, delta_nabla_w = backprop(Net, x, y)
		nabla_b = [nb+dnb for (nb,dnb) = zip(nabla_b, delta_nabla_b)]
		nabla_w = [nw+dnw for (nw,dnw) = zip(nabla_w, delta_nabla_w)]
	end
	Net.biases = [b - eta/M * nb for (b,nb) = zip(Net.biases, nabla_b)]
	Net.weights = [w - eta/M * nw for (w,nw) = zip(Net.weights, nabla_w)]
end

function backprop(Net::Network, x, y)
	nabla_b = [zeros(size(b)) for b in Net.biases]
	nabla_w = [zeros(size(w)) for w in Net.weights]
	activation = x
	activations = [x]
	zs = Any[]
	for (b,w) = zip(Net.biases, Net.weights)
		z = w*activation + b
		println(typeof(activation))
		println(typeof(z))
		append!(zs, [z])
		activation = sigmoid(z)
		append!(activations, [activation])
	end
	delta = cost_derivative(activations[end], y) .* sigmoid_prime(zs[end])
	nabla_b[end] = delta
	nabla_w[end] = delta * activations[end-1]'
	for l = 1:Net.Nlayers-1
		z = zs[end-l]
		sp = sigmoid_prime(z)
		delta = Net.weights[-l+1]' * delta * sp
		nabla_b[end-l] = delta
		nabla_w[end-l] = delta * activations[end-l-1]'
	end
	return nabla_b, nabla_w
end

cost_derivative(output, true_output) = output - true_output


end #module
