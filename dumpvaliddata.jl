#!/bin/bash
# -*- mode: julia -*-
#=
exec julia --project --color=yes --startup-file=no -e 'include(popfirst!(ARGS))' \
"${BASH_SOURCE[0]}" "$@"
=#

include(joinpath(@__DIR__, "train_pipeline.jl"))

dumpdata(ARGS)
