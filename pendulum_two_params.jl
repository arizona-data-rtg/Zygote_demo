"""
In this file we develop a differentiable programming scheme for learnig physical
  paramters and unknown functions from data for the pendulum problem

extended to two paramters
(next can we learn sin function?)
∂_t x = v, ∂_t v = -sinx - αv + F
"""


using Plots, Zygote

N = 1000;
h = 1/N;
Y = [0., 3., 0., 0.];
x = Y[1:2]
println(x)
println(Y)

function f(x, F, α)
    return [x[2], - sin(x[1]) - α * x[2] + F]
end


#RK4 method
function RK4(f, x, F, α)
    for n in 1 : N
        k1 = h * f(x, F, α);
        k2 = h * f(x + 0.5 * k1, F, α);
        k3 = h * f(x + 0.5 * k2, F, α);
        k4 = h * f(x + k3, F, α);
        x += (k1 + 2. * k2 + 2. * k3 + k4)/6
    end
    return x
end


function l2_loss(f, x, F, α, y_gt)
    y_pred = RK4(f, x, F, α)
    return sum((y_pred .- y_gt).^2)
end


F_gt = 0.1;
α_gt = 0.2;

y_gt = RK4(f, x, F_gt, α_gt)
display(y_gt)

F = 0.2 * rand()
α = 0.4 * rand()
y_pred = RK4(f, x, F, α)

θ = (0.4, 0.35)
∂_θ = gradient(θ -> l2_loss(f, x, θ[1], θ[2], y_gt), θ)[1]
println("∂l_∂θ = ", ∂_θ)
# display(∂_F)



function training_autograd(n_epchs, sample_rate, f, x, y_gt)
    F_hat = 0.40
    α_hat = 0.35
    θ = (F_hat, α_hat)
    F_track = zeros(round(Int, n_epchs/sample_rate))
    α_track = zeros(round(Int, n_epchs/sample_rate))
    lr_F = 5e-2
    lr_α = 5e-2
    lr = (lr_F, lr_α)
    ii = 1;
    for k ∈ 1 : n_epchs
        ∂_θ = gradient(θ -> l2_loss(f, x, θ[1], θ[2], y_gt), θ)[1]
        θ = θ .- lr .* ∂_θ
        if mod(k, sample_rate) == 0
            F_track[ii] = θ[1]
            α_track[ii] = θ[2]
            ii += 1;
            println("iteration  ", k, "  F_hat = ", θ[1], "  α_hat = ", θ[2])
        end
    end
    return F_track, α_track
end



sample_rate = 1;
n_epchs = 35000

F_out, α_out = training_autograd(n_epchs, sample_rate, f, x, y_gt)
F_gt_data = F_gt * ones(size(F_out))
α_gt_data = α_gt * ones(size(α_out))

# println(F_out)
plt = plot(F_out, label="F_hat", color="green")
plot!(F_gt_data, label="F_gt", color="green", linestyle=:dash)
plot!(α_out, label="α_hat", color="blue")
plot!(α_gt_data, label="α_gt", color="blue", linestyle=:dash)
title!("learning F, α with autograd")
xlabel!("epochs")
ylabel!("parameters")
display(plt)
savefig(plt, "diffp_learning_two_params3.png")



























#-------Analytic method

# function ℱ(x, F)
#     return [x[2], -sin(x[1]) - 0.2 * x[2] + F,
#             x[4], -cos(x[1]) * x[3] - 0.2 * x[4] + 1.];
# end
#

    # function ∂x_loss(x, v)
    #     return x/sqrt(x^2 + v^2)
    # end
    #
    # function ∂v_loss(x, v)
    #     return v/sqrt(x^2 + v^2)
    # end
    #
    #
    # function training_algorithm(n_epchs, sample_rate, x_gt, v_gt, lr, Y, ℱ)
    #     F_hat = rand()
    #     F_track = zeros(round(Int, n_epchs/sample_rate))
    #     ii = 1;
    #     for k ∈ 1 : n_epchs
    #         Y_hat = RK4(ℱ, Y, F_hat)
    #         x_hat = Y_hat[1]
    #         v_hat = Y_hat[2]
    #         h1_hat = Y_hat[3]
    #         h2_hat = Y_hat[4]
    #         gradxL = ∂x_loss(x_hat - x_gt, v_hat - v_gt) * h1_hat
    #         gradvL = ∂v_loss(x_hat - x_gt, v_hat - v_gt) * h2_hat
    #         F_hat = F_hat - lr * (gradxL + gradvL)
    #         if mod(k, sample_rate) == 0
    #             F_track[ii] = F_hat
    #             ii += 1;
    #         end
    #     end
    #     return F_track
    # end
    #
    # F_gt = 0.1
    # xv = RK4(f, x, F_gt)
    # x_gt = xv[1]
    # v_gt = xv[2];
    # println(x_gt, " ", v_gt)
    #
    # n_epchs = 4000;
    # sample_rate = 1;
    # lr = 1e-5
    #
    # F_out = training_algorithm(n_epchs, sample_rate, x_gt, v_gt, lr, Y, ℱ)
    # F_gt_data = F_gt * ones(size(F_out))
    # # println(F_out)
    # plt = plot(F_out, label="F_hat", color="green")
    # plot!(F_gt_data, label="F_gt", color="green", linestyle=:dash)
    # title!("learning external forcing F")
    # xlabel!("epochs")
    # ylabel!("F_hat")
    # display(plt)























#--------------------Tests


# F = 0.1113
# xv = RK4(f, x, F)
# println(xv)

# K = 10000
# xv_track = zeros(K, 2)
#
#
# F = -0.112
# dF = 0.00001
# for k in 1 : K
#     global F
#     xv_track[k, :] .= RK4(x, F)
#     F += dF
# end
#
# plot_xv(xv_track[:, 1], xv_track[:, 2])


# function plot_xv(x, v)
#     display(scatter(x, v))
# end
#
