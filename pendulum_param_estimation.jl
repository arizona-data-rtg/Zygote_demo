"""
In this file we develop a differentiable programming scheme for learnig physical
  paramters and unknown functions from data for the pendulum problem

∂_t x = v, ∂_t v = -sinx - 1/5v + F = f

can extend to many paramters and finding unknown functions (NNs)
∂_t x = v, ∂_t v = -sinx - αv + F
"""

using Plots, Zygote

N = 3000;
h = 3/N;
x = [0., 3.];


function f(x, F)
    return [x[2], -sin(x[1]) - 0.2*x[2] + F]
end


#RK4 method (Higher order function)
function RK4(f, x, F)
    for n in 1 : N
        k1 = h * f(x, F);
        k2 = h * f(x + 0.5 * k1, F);
        k3 = h * f(x + 0.5 * k2, F);
        k4 = h * f(x + k3, F);
        x += (k1 + 2. * k2 + 2. * k3 + k4)/6
    end
    return x
end


function l2_loss(f, x, F, y_gt)
    y_pred = RK4(f, x, F)
    return sum((y_pred .- y_gt).^2)
end


F_gt = 0.1;
y_gt = RK4(f, x, F_gt)
display(y_gt)



function training_autograd(n_epchs, sample_rate, f, x, y_gt)
    F_hat = rand()
    F_track = zeros(round(Int, n_epchs/sample_rate))
    lr = 1e-3
    ii = 1;
    for k ∈ 1 : n_epchs
        ∂_F = gradient(F_hat -> l2_loss(f, x, F_hat, y_gt), F_hat)[1]
        F_hat = F_hat - lr * ∂_F
        if mod(k, sample_rate) == 0
            F_track[ii] = F_hat
            ii += 1;
            println("iteration  ", k, "  F_hat = ", F_hat)
        end
    end
    return F_track
end



sample_rate = 1;
n_epchs = 300

F_out = training_autograd(n_epchs, sample_rate, f, x, y_gt)
F_gt_data = F_gt * ones(size(F_out))
# println(F_out)
plt = plot(F_out, label="F_hat", color="green")
plot!(F_gt_data, label="F_gt", color="green", linestyle=:dash)
title!("learning external forcing F with autograd")
xlabel!("epochs")
ylabel!("F_hat")
savefig(plt, "diffp_learning_F.png")
display(plt)
