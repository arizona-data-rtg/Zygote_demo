"""
In this file we develop the analytic and diff_prog method for the
pendulum problem taking advantage of the julia ODE package to simultaneous
integration of forward sensitivity problem

Goal:   -- mix automatic differenation and analytic differentation

        (CHECK) -- incorparate Flux training package to use ADAM() knowing gradient.

        -- Utilize stochastc gradient descent by batching


Two parameters
        ∂_t x = v, ∂_t v = -sinx - αv + F

        NOTES:
            Working for N = 2000, h = 1/N, lr = 1e-3, n_epchs = 35000
"""



# include("./plots.jl")

using DifferentialEquations, Plots, Flux
using Flux.Optimise: update!

tspan = (0.0, 1.0)
T = 1.0
dt = T/2000
tsteps = 0.0:dt:T

# Initial condition
z0 = zeros(6);
z0[1:2] = [0., 3.0]

function pendulum_H!(dz, z, p, t)
  α, F = p
  x = z[1]; v = z[2]; hα = z[3:4]; hF = z[5:6];
  ∂F_∂Y = [0  1; -cos(x)  -α];
  ∂F_∂α = [0; -v]; ∂F_∂F = [0; 1];
  dz[1] = v
  dz[2] = -sin(x) - α*v + F
  dz[3:4] = ∂F_∂Y * hα + ∂F_∂α
  dz[5:6] = ∂F_∂Y * hF + ∂F_∂F
  return dz
end


p_gt = [0.2, 0.1]
prob = ODEProblem(pendulum_H!, z0, tspan, p_gt)
sol_gt = solve(prob, Tsit5(), p=p_gt, saveat = tsteps)


function loss(sol_pred, sol)
  return sum((sol_pred[1,:] - sol[1,:]).^2 + (sol_pred[2,:] - sol[2,:]).^2)  #same as mse
end

function ∇L(pred, sol, index)
    return 2*sum((pred[1,:] .- sol[1,:]) .* pred[index, :] + (pred[2,:] .- sol[2,:]) .* pred[index + 1, :])
end

lr = 1e-2
opt = ADAM(lr)
function training_algorithm(n_epchs, sample_rate, sol, lr)
      p_hat = [2.0, 1.95];
      F_track = zeros(round(Int, n_epchs/sample_rate))
      α_track = zeros(round(Int, n_epchs/sample_rate))
      ii = 1;
      for k ∈ 1 : n_epchs
          sol_pred = solve(prob, Tsit5(), p=p_hat, saveat = tsteps)
          ∇L_v = [∇L(sol_pred, sol, 3), ∇L(sol_pred, sol, 5)]
          update!(opt, p_hat, ∇L_v)
          if mod(k, sample_rate) == 0
              α_track[ii] = p_hat[1]
              F_track[ii] = p_hat[2]
              ii += 1;
          end
          if mod(k, 100) == 0
              gradL = ∇L(sol_pred, sol, 3) + ∇L(sol_pred, sol, 5)
              println("iteration = ", k, "   α_hat = ", p_hat[1], "   F_hat = ", p_hat[2], "   gradL = ", gradL )
          end
      end
      return F_track, α_track
end


n_epchs = 40000;
sample_rate = 1;

F_out, α_out = training_algorithm(n_epchs, sample_rate, sol_gt, lr)
F_gt_data = p_gt[2] * ones(size(F_out))
α_gt_data = p_gt[1] * ones(size(F_out))
theme(:default)
# gen_plots()
