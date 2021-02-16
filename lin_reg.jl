using Zygote, LinearAlgebra

# LinearRegression object, containing multiple fields, some of which will be learned.
mutable struct LinearRegression
    # These values will be implicitly learned
    weights::Matrix
    bias::Float64

    # These values will not be learned
    name::String
end

#Function for building LinearRegression object initializing fields
LinearRegression(nparams, name) = LinearRegression(randn(1, nparams), 0.0, name)



# y = w*X + b
function predict(model::LinearRegression, X)
    return model.weights * X .+ model.bias
end

#L2
function loss(model::LinearRegression, X, Y)
    return norm(predict(model, X) .- Y, 2)
end


# Our "ground truth" values
weights_gt = [1.0, 2.7, 0.3, 1.2]'
bias_gt = 0.4

# Generate a dataset of 10,000 observations
X = randn(length(weights_gt), 10000)
Y = weights_gt * X .+ bias_gt

# Add noise to `X`
X .+= 0.001.*randn(size(X))

model = LinearRegression(size(X, 1), "Example")

println("pre-trained loss = ", loss(model::LinearRegression, X, Y))


# Calculate gradient upon `model` for the first example in our training set
grads = Zygote.gradient(model) do m
    return loss(m, X[:,1], Y[1])
end

display(grads)
# Tuple containing one element per argument to gradient(). The first element gets the gradient upon the
# model.

grads = grads[1]

display(grads)
#contains the ∂L/∂θ (sensitivity) values

grads = grads[]

display(grads)

display(grads.weights)


#Here we will just use the simplest standard gradient descent algorithm:
function grad_descent_update!(model::LinearRegression, grads, η = 0.001)
    model.weights .-= η .* grads.weights
    model.bias -= η * grads.bias
end



@info("Running train loop for $(size(X,2)) iterations")
for idx in 1 : size(X, 2)
    grads = Zygote.gradient(m -> loss(m, X[:, idx], Y[idx]), model)[1][]
    grad_descent_update!(model, grads)
end

println("model weights = ", model.weights)
println("   weights_gt = ", weights_gt)

println("

post-trained loss = ", loss(model::LinearRegression, X, Y))
