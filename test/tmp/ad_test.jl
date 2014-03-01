using AutoDiff
using ODE
using Benchmark

function F(t, y)
	[-y[2], y[1] ]
end
const y_0=[1.0, 0.0]
const y_0d=ad(y_0)
tspan=[0.0, 1.0]

t,A=ode45(F, tspan, y_0)
td,Ad=ode45(F, tspan, y_0d)

ode_normal() = ode45(F, tspan, y_0)
ode_ad() = ode45(F, tspan, y_0d)

print(compare([ode_normal, ode_ad], 1000))
