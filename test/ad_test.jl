using ODE

function F(t, y)
	[-y[2], y[1] ]
end
const y_0=[1.0, 0.0]
const y_0d=ad(y_0)
tspan=[0.0, 1.0]

t,A=ode45(F, tspan, y_0)
td,Ad=ode45(F, tspan, y_0d)

@time for i=1:1000;t,A=ode45(F, tspan, y_0);end
@time for i=1:1000;td,Ad=ode45(F, tspan, y_0d);end

