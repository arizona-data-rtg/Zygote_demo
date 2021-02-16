"""
Exploring differentiable programming with Zygote. Claims:
    control flow
    recursion
    nesting
    mutation
    data structures
    higher order functions
    ...

See "Don't Unroll Adjoint: Differentiating SSA-form programs" for the understanding the magic
"""


using Zygote, Plots

# truncated taylor series for sin with
# control flow, and recursion:
function s(x)
    ts = 0.0
    sign = -1.0
    for i in 1:19 #Truncated at 10th term
        if isodd(i)
            newterm = x^i/factorial(i)
            abs(newterm)<1e-8 && return ts
            # println("i=",i)
            sign = -sign
            ts += sign * newterm
        end
    end
    return ts
end

c = Zygote.gradient(s, 0.5)  #∂s/∂x(0.5)
display(c) #returns tuple
c = c[1]   #first component gives the derivative
c_t = cos(0.5)
println("  c = ", c, "
c_t = ", c_t)


∇s(x) = gradient(s, x)[1]
display(∇s)
println(cos(1.0))
println(∇s(1.0))

p1 = plot(x -> cos(x), -2*pi, 2*pi)
scatter!(x -> ∇s(x), -2*pi, 2*pi)
display(p1)




f(x) = mod(x, 2*pi)
∇f(x) = gradient(f, x)[1]
p2 = plot(x -> f(x), 0, 8*pi)
plot!(x -> ∇f(x), 0, 8*pi)
display(p2)




#Linear transformation
A = rand(2, 3); x = rand(3);
ga = gradient(A -> sum(A*x), A)[1]
display(ga)


#Recursion and control flow
function pow(x, n)
    r = 1
    for i = 1:n
      r *= x
    end
    return r
end
println(gradient(x -> pow(x, 3), 2))



#-------mutable data structures are supported:
d = Dict()
display(d)

gradient(5) do x
    d[:x] = x
    d[:x] * d[:x]
end

#here is essientally what the do block does:
f1x = map(x -> 2*x, 1:10)

fdx = map(1:10) do x
    2x
end
println(f1x)
println(fdx)



#------------------Piecwise (cubic spline)

h = 0.5
sigma = (10. / (7. * pi * h * h));

function W(r)
  q = r / h;
  if (q > 2.)
    return 0.;
  end
  if (q > 1.)
    return (sigma * (2. - q)^3 / 4.);
  end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

pw = plot(x -> W(x), 0, 2*h)
display(pw)


# exact
function dW_dr(r)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h));   end
  return (sigma * r * (-3. + 9. * q / 4.) / h^2);
end

pdw = plot(x -> dW_dr(x), 0, 2*h)
display(pdw)


∇W(x) = Zygote.gradient(W, x)[1]

p3 = scatter(x -> ∇W(x), 0, 2*h)
plot!(x -> dW_dr(x), 0, 2*h)
display(p3)




#----multi variate functions


g1(a,b) = gradient((a, b) -> a*b, a, b)


function ff(x, y)
    return 3*x^2 + 1 + y^3 - 3*y
end

g2(x,y) = gradient(ff, x, y)[1]
#returns ∂f/∂x









# function W2(x, y)
#   r = sqrt(x^2 + y^2)
#   q = r / h;
#   if (q > 2.)
#     return 0.;
#   end
#   if (q >= 1.)
#     return (sigma * (2. - q)^3 / 4.);
#   end
#   return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
# end
#
# # exact
# function dW_dr2(x, y)
#   r = sqrt(x^2 + y^2)
#   q = r / h;   if (q > 2.)   return 0.;   end
#   if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h));   end
#   return (sigma * r * (-3. + 9. * q / 4.) / h^2);
# end
#
#
# x_grid2 = -2*h:0.005:2*h
# y_grid2 = -2*h:0.005:2*h
#
# pw2 = plot(x_grid2, y_grid2, W2, st=:surface)
# display(pw2)
# dW2(x, y) = gradient(W2, x, y)
#
# # pdw2 = plot(x_grid2./1.5, y_grid2./1.5, dW2, st=:surface)
# # display(pdw2)
#
# p_ex_dw2 = plot(x_grid2./1.5, y_grid2./1.5, dW_dr2, st=:surface)
# display(p_ex_dw2)
# # dw2 = dW2(x_grid2, y_grid2)
