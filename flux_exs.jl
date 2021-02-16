"""
Flux uses Zygote (and other packages) for ML applications
"""

using Flux, Plots

f(x) = 3x^2 + 2x + 1;
∂f(x) = 6x + 2;

df(x) = gradient(f, x)[1];


x_data = -1. : 0.1 : 1.
f_data = f.(x_data)
∂f_data = ∂f.(x_data)
df_data = df.(x_data)

plot!(x_data, ∂f_data, label="∂f")
scatter!(x_data, df_data, label="Δf")

display(p1)




#---------Linear regression

A = randn(2, 5);
b = randn(2);

lin_model(x) = A*x .+ b

function l2_loss(x, y)
    y_hat = lin_model(x)
    return sum((y .- y_hat).^2)
end

x, y = randn(5), randn(2)

display(A)
gs = gradient(() -> l2_loss(x, y), params(A,b))
display(gs[A])


epochs = 200
lr = 0.05
for k in 1:epochs
    global A, b
    gs = gradient(() -> l2_loss(x, y), params(A,b))
    A_grad = gs[A];
    b_grad = gs[b];

    A = A - lr .* A_grad
    b = b - lr .* b_grad

    println("loss = ", l2_loss(x,y))
end

display(A)

#-------------Simple NN
