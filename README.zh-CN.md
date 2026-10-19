[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/JuliaDiff/ForwardDiff.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/JuliaDiff/ForwardDiff.jl/actions/workflows/ci.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/JuliaDiff/ForwardDiff.jl/graph/badge.svg?token=SzHfhyoxUa)](https://codecov.io/gh/JuliaDiff/ForwardDiff.jl)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/ForwardDiff.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/ForwardDiff.jl/dev)

# ForwardDiff.jl

> 本文是 [README.md](README.md) 的中文翻译。两者如有出入，以英文原文为准。

## 升级到 ForwardDiff.jl 1.0

`Dual` 数上的相等比较（`==`）现在要求实部和对偶部都相同（https://github.com/JuliaDiff/ForwardDiff.jl/pull/481）。
这修复了一大批 bug：这些 bug 源于程序会去检查例如数组中非零值之类的“结构”，从而得到错误的导数。
这可能会让程序行为略有变化，但总体上比之前更正确。

---------------------------

ForwardDiff 使用**前向模式自动微分（AD）**，对原生 Julia 函数（实际上对任何可调用对象都适用）计算**导数**、**梯度**、**雅可比矩阵**、**黑塞矩阵**以及更高阶导数。

性能会随所计算的函数而变化，但总体而言，ForwardDiff 实现的算法在速度和精度上都优于非 AD 算法（例如有限差分）。

下面是一个简单示例：

```julia
julia> using ForwardDiff

julia> f(x::Vector) = sin(x[1]) + prod(x[2:end]);  # returns a scalar

julia> x = vcat(pi/4, 2:4)
4-element Vector{Float64}:
 0.7853981633974483
 2.0
 3.0
 4.0

julia> ForwardDiff.gradient(f, x)
4-element Vector{Float64}:
  0.7071067811865476
 12.0
  8.0
  6.0

julia> ForwardDiff.hessian(f, x)
4×4 Matrix{Float64}:
 -0.707107  0.0  0.0  0.0
  0.0       0.0  4.0  3.0
  0.0       4.0  0.0  2.0
  0.0       3.0  2.0  0.0
```

像 `f` 这样把向量映射为标量的函数，是反向模式自动微分的最佳场景；但如果 `x` 不太大，ForwardDiff 可能仍是更好的选择，因为它简单得多。
前向模式微分的最佳场景是把标量映射为向量的函数，例如下面的 `g`：

```julia
julia> g(y::Real) = [sin(y), cos(y), tan(y)];  # returns a vector

julia> ForwardDiff.derivative(g, pi/4)
3-element Vector{Float64}:
  0.7071067811865476
 -0.7071067811865475
  1.9999999999999998

julia> ForwardDiff.jacobian(x) do x  # anonymous function, returns a length-2 vector
         [sin(x[1]), prod(x[2:end])]
       end
2×4 Matrix{Float64}:
 0.707107   0.0  0.0  0.0
 0.0       12.0  8.0  6.0
```

使用说明详见 [ForwardDiff 文档](https://juliadiff.org/ForwardDiff.jl/stable)。
许多简单函数（例如 `sin`）的导数由 [DiffRules](https://github.com/JuliaDiff/DiffRules.jl) 提供。

其他自动微分包见 [JuliaDiff 主页](https://juliadiff.org)。

## 论文与引用

如果 ForwardDiff 对你的工作有帮助，我们恳请你引用[这篇论文](https://arxiv.org/abs/1607.07892)：

```bibtex
@article{RevelsLubinPapamarkou2016,
    title = {Forward-Mode Automatic Differentiation in {J}ulia},
   author = {{Revels}, J. and {Lubin}, M. and {Papamarkou}, T.},
  journal = {arXiv:1607.07892 [cs.MS]},
     year = {2016},
      url = {https://arxiv.org/abs/1607.07892}
}
```
