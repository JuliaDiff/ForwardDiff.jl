# Functions for Bayesian logit model with a Normal prior N(0, priorVar*I)

function logPrior(pars, nPars::Int, data::Dict{Any, Any})
  return (-dot(pars,pars)/data["priorVar"]
    -nPars*log(2*pi*data["priorVar"]))/2
end

function logLikelihood(pars, nPars::Int, data::Dict{Any, Any})
  XPars = data["X"]*pars
  return (XPars'*data["y"]-sum(log(1+exp(XPars))))[1]
end

function gradLogPosterior(pars::Vector{Float64}, nPars::Int,
  data::Dict{Any, Any})
  adPars = gradual(pars)
  return grad(logLikelihood(adPars, nPars, data)+logPrior(adPars, nPars, data))
end

function randPrior(nPars::Int, data::Dict{Any, Any})
  return rand(Normal(0.0, sqrt(data["priorVar"])), nPars)
end
