#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

include(joinpath(@__DIR__, "train_pipeline.jl"))

train(ARGS)
