using Random
using LinearAlgebra

using TrajectoryOptimization
const TO = TrajectoryOptimization
using Flux
using JLD2


using StructuredMechModels
const SMM = StructuredMechModels

include(joinpath(@__DIR__, "datautils.jl"))
include(joinpath(@__DIR__, "tasks.jl"))


function run_experiment(seed, mtype, taskname, ntrajs::Int, traj_len::Int, hparams)
    Random.seed!(seed)
    params = TASK_REGISTRY[taskname]
    task = params.sys
    thetamask = params.thetamask
    defhparams = params.hparams
    task_d = discretize_model(task, :rk4)
    hparams = merge(defhparams, hparams)
    if !hparams.wraptopi
        thetamask = zero(thetamask)
    end
    env = SMM.Env(task_d, randn(task_d.n), thetamask)

    @info "Generating data..."
    if hparams.dataprocess == "randn"
        train_dataset = generate_rand_data(env, ntrajs, traj_len, hparams.dt, maxu=hparams.maxu, stddev=hparams.stddev)
    elseif hparams.dataprocess == "sine"
        train_dataset = generate_sine_data(env, ntrajs, traj_len, hparams.dt, maxu=hparams.maxu)
    else
        error("Dataprocess: $(hparams.dataprocess) not defined")
    end

    train_dataset = SMM.addnoise(train_dataset, hparams.noisestd)
    valid_dataset_path = joinpath(hparams.logdir, taskname, "validation_dataset_dt=$(hparams.dt).jld2")
    JLD2.@load valid_dataset_path valid_dataset

    qdim = round(Int, task_d.n/2)
    logdir = joinpath(hparams.logdir, taskname, mtype,
                      "dataprocess=$(hparams.dataprocess)", "ntrajs=$ntrajs",
                      "seed=$seed,traj_len=$traj_len,dt=$(hparams.dt)_wraptopi=$(hparams.wraptopi)")

    nn_jl = getfield(SMM, Symbol(mtype))(qdim, task_d.m, thetamask; hidden_sizes=hparams.hidden_sizes[mtype])
    nn_py = getfield(SMM.Torch, Symbol(mtype))(qdim, task_d.m, thetamask; hidden_sizes=hparams.hidden_sizes[mtype])
    hparams = merge(hparams, (logdir=logdir,))

    @info "Training..."
    SMM.Torch.train!(nn_py, train_dataset, valid_dataset, hparams)
    Flux.loadparams!(nn_jl, SMM.Torch.params(nn_py))

    weights = Flux.params(nn_jl)
    JLD2.@save joinpath(logdir, "final_best_ckpt.jld2") weights
    @info "Done!"
end

function train(args)
    if length(args) != 7
        error("Usage: ./train.jl <model type> <task name> <ntrajs> <seed> <nepochs> <batch size> <logdir>")
    end
    mtype = args[1]
    taskname = args[2]
    ntrajs = parse(Int, args[3])
    seed = parse(Int, args[4])
    nepochs = parse(Int, args[5])
    batch_size = parse(Int, args[6])
    logdir = args[7]
    dt = 0.05
    traj_len = 3

    hparams=(scheduler_step_size=round(Int, nepochs/2),
             dataprocess="randn",
             wraptopi=true,
             nepochs=nepochs,
             dt=dt,
             batch_size=batch_size,
             logdir=logdir)

    run_experiment(seed, mtype, taskname, ntrajs, traj_len, hparams)
end

function dumpdata(args)
    Random.seed!(42)

    taskname = args[1]
    outdir = args[2]

    ntrajs = 256*64
    traj_len = 3
    dt = 0.05
    if taskname == "doublecartpole"
        ntrajs *= 2
    end

    params = TASK_REGISTRY[taskname]
    task = params.sys
    thetamask = params.thetamask
    defhparams = params.hparams
    task_d = discretize_model(task, :rk4)

    env = SMM.Env(task_d, randn(task_d.n), thetamask)
    valid_dataset = generate_rand_data(env, ntrajs, traj_len, dt; maxu=defhparams.maxu, stddev=defhparams.stddev)
    path = joinpath(outdir, taskname)
    mkpath(path)
    JLD2.@save joinpath(path, "validation_dataset_dt=$dt.jld2") valid_dataset
end
