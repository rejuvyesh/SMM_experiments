#!/usr/bin/env bash
# -*- mode: julia -*-
#=
exec julia --project --color=yes --startup-file=no -e 'include(popfirst!(ARGS))' \
"${BASH_SOURCE[0]}" "$@"
=#

include(joinpath(@__DIR__, "eval_pipeline.jl"))

evaluate(ARGS)
