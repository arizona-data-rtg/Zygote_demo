"""
In this file we develop a differentiable programming scheme for learnig physical
  paramters and unknown functions from data for the pendulum problem

∂_t x = v, ∂_t v = -sinx - αv + 0.1 = f

"""

using Plots, Zygote

N = 3000;
h = 3/N; #3s of integration with 1000 steps per second
x = [0., 3.];

function f(x, α)
    return [x[2], -sin(x[1]) - α*x[2] + 0.1]
end


#RK4 method
function RK4(f, x, α)
    for n in 1 : N
        k1 = h * f(x, α);
        k2 = h * f(x + 0.5 * k1, α);
        k3 = h * f(x + 0.5 * k2, α);
        k4 = h * f(x + k3, α);
        x += (k1 + 2. * k2 + 2. * k3 + k4)/6
    end
    return x
end


function l2_loss(f, x, α, y_gt)
    y_pred = RK4(f, x, α)
    return sum((y_pred .- y_gt).^2)
end


#Obtain "synthetic" ground truth data
α_gt = 0.1;
y_gt = RK4(f, x, α_gt) #obains final angle and velocity at t = 3s;


function training_autograd(n_epchs, lr, f, x, y_gt)
    α_hat = 0.9 #initial guess
    α_track = zeros(round(Int, n_epchs))
    ii = 1;
    for k ∈ 1 : n_epchs
        ∂L_α = gradient(α_hat -> l2_loss(f, x, α_hat, y_gt), α_hat)[1]
        α_hat = α_hat - lr * ∂L_α
        α_track[ii] = α_hat
        ii += 1;
        println("iteration  ", k, "  α_hat = ", α_hat)
    end
    return α_track
end



sample_rate = 1;
n_epchs = 40
lr = 1e-3


α_out = training_autograd(n_epchs, lr, f, x, y_gt)
α_gt_data = α_gt * ones(size(α_out))

plt = plot(α_out, label="α_hat", color="green")
plot!(α_gt_data, label="α_gt", color="green", linestyle=:dash)
title!("Learning diffusion parameter with autograd")
xlabel!("epochs")
ylabel!("α_hat")
savefig(plt, "diffp_learning_α.png")
display(plt)
