using Distributions

using StructuredMechModels

using StructuredMechModels: Env, batch_rollout, trajbatch2dataset

const SIN_PARAMS = (0.6, [(amp=0.5, phase=0.0, freq=1.031),
                          (amp=0.4, phase=0.7, freq=2.424),
                          (amp=0.1, phase=0.4, freq=5.916)])

function sine_offsetsignal(B, T, dt, udim; params=SIN_PARAMS)
    t_points = repeat(collect(0:dt:T*dt)[1:T-1], 1, udim)# TODO
    offset, argss = params
    us = zeros(Float64, udim, T-1, B)
    for b in 1:B
        arg = rand(argss)
        us[:, :, b] .= (arg.amp .* sin.(arg.freq .* t_points .+ arg.phase) .+ offset)'
    end
    return us
end


function generate_rand_data(env::Env, ntrajs, traj_len, dt; maxu=100.0, stddev=30.0)
    @assert traj_len > 1
    @assert ntrajs >= 1
    init_xs = rand(Uniform(-2π, 2π), env.sys.n, ntrajs)
    us = clamp.(stddev * randn(env.sys.m, traj_len-1, ntrajs), -maxu, maxu)
    xs, us = batch_rollout(env, init_xs, us, traj_len, ntrajs, dt)
    return trajbatch2dataset(xs, us)
end

function generate_sine_data(env::Env, ntrajs, traj_len, dt; maxu=100.0)
    @assert traj_len > 1
    @assert ntrajs >= 1
    init_xs = rand(Uniform(-2π, 2π), env.sys.n, ntrajs)
    us = sine_offsetsignal(ntrajs, traj_len, dt, env.sys.m)
    us = clamp.(us)
    xs, us = batch_rollout(env, init_xs, us, traj_len, ntrajs, dt)
    return trajbatch2dataset(xs, us)
end

