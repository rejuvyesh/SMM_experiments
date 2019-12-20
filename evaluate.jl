#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

include(joinpath(@__DIR__, "eval_pipeline.jl"))

evaluate(ARGS)
