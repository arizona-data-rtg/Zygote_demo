"""
See DiffEqFlux for more info
"""

using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat = tsteps)

# Plot the solution
psol = plot(sol, title = "Lotka Volterra")
display(psol)

sol_pred = solve(prob, Tsit5(), p=p, saveat = tsteps)


# for find parameters that generated gt
function loss(p)
  # prob2 = ODEProblem(lotka_volterra!, u0, tspan, p)
  sol_pred = solve(prob, Tsit5(), p=p, saveat = tsteps)
  loss = sum(abs2, sol_pred .- sol)
  return loss, sol_pred
end

# display learning
callback = function (p, l, pred)
  display(l)
  plt = plot(sol)
  scatter!(pred, ylim = (0, 7))
  display(plt)
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  sleep(1.0)
  return false
end


p = [1.8, 0.8, 3.1, 1.1];

# parameter estimation
result_ode = DiffEqFlux.sciml_train(loss, p,
                                    ADAM(0.1),
                                    cb = callback,
                                    maxiters = 100)
