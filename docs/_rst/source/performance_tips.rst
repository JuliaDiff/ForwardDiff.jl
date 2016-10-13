Performance Tips
================

SIMD Vectorization
------------------

Many operations on ForwardDiff's dual numbers are amenable to `SIMD vectorization
<https://en.wikipedia.org/wiki/SIMD#Hardware>`_. For some ForwardDiff benchmarks, we've
seen SIMD vectorization yield `speedups of almost 3x
<https://github.com/JuliaDiff/ForwardDiff.jl/issues/98#issuecomment-253149761>`_.

To enable SIMD optimizations, start your Julia process with the ``-O3`` flag. This flag
enables `LLVM's SLPVectorizerPass
<http://llvm.org/docs/Vectorizers.html#the-slp-vectorizer>`_ during compilation, which
attempts to automatically insert SIMD instructions where possible for certain arithmetic
operations.

Here's an example of LLVM bitcode generated for an addition of two ``Dual`` numbers without
SIMD instructions (i.e. not starting Julia with ``-O3``):

.. code-block:: julia

    julia> using ForwardDiff: Dual

    julia> a = Dual(1., 2., 3., 4.)
    Dual(1.0,2.0,3.0,4.0)

    julia> b = Dual(5., 6., 7., 8.)
    Dual(5.0,6.0,7.0,8.0)

    julia> @code_llvm a + b

    define void @"julia_+_70852"(%Dual* noalias sret, %Dual*, %Dual*) #0 {
    top:
      %3 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 0
      %4 = load double, double* %3, align 8
      %5 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 0
      %6 = load double, double* %5, align 8
      %7 = fadd double %4, %6
      %8 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 1
      %9 = load double, double* %8, align 8
      %10 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 1
      %11 = load double, double* %10, align 8
      %12 = fadd double %9, %11
      %13 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 2
      %14 = load double, double* %13, align 8
      %15 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 2
      %16 = load double, double* %15, align 8
      %17 = fadd double %14, %16
      %18 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 0
      %19 = load double, double* %18, align 8
      %20 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 0
      %21 = load double, double* %20, align 8
      %22 = fadd double %19, %21
      %23 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 0
      store double %22, double* %23, align 8
      %24 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 0
      store double %7, double* %24, align 8
      %25 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 1
      store double %12, double* %25, align 8
      %26 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 2
      store double %17, double* %26, align 8
      ret void
    }

If we start up Julia with ``-O3`` instead, the call to ``@code_llvm`` will show that LLVM
can SIMD-vectorize the addition:

.. code-block:: julia

    julia> @code_llvm a + b

    define void @"julia_+_70842"(%Dual* noalias sret, %Dual*, %Dual*) #0 {
    top:
      %3 = bitcast %Dual* %1 to <4 x double>*            # cast the Dual to a SIMD-able LLVM vector
      %4 = load <4 x double>, <4 x double>* %3, align 8
      %5 = bitcast %Dual* %2 to <4 x double>*
      %6 = load <4 x double>, <4 x double>* %5, align 8
      %7 = fadd <4 x double> %4, %6                      # SIMD add
      %8 = bitcast %Dual* %0 to <4 x double>*
      store <4 x double> %7, <4 x double>* %8, align 8
      ret void
    }

Note that whether or not SIMD instructions can actually be used will depend on your machine.
