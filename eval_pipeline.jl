using LinearAlgebra
using JSON
using Statistics
using ProgressMeter
using JLD2
using Flux
using TrajectoryOptimization
const TO = TrajectoryOptimization

using StructuredMechModels
const SMM = StructuredMechModels

include(joinpath(@__DIR__, "tasks.jl"))
include(joinpath(@__DIR__, "evalutils.jl"))


function run_cost_evaluation(mtype, taskname, loadpath, dt; viz=false)
    params = TASK_REGISTRY[taskname]

    task_d = discretize_model(params.sys, :rk4)
    env = SMM.Env(task_d, randn(task_d.n), params.thetamask)
    qdim = round(Int, task_d.n/2)

    hidden_sizes = params.hparams.hidden_sizes[mtype]
    nn_jl = getfield(SMM, Symbol(mtype))(qdim, task_d.m, params.thetamask; hidden_sizes=hidden_sizes)
    nn_py = getfield(SMM.Torch, Symbol(mtype))(qdim, task_d.m, params.thetamask; hidden_sizes=hidden_sizes)
    SMM.Torch.load_state!(nn_py, loadpath)
    Flux.loadparams!(nn_jl, SMM.Torch.params(nn_py))
    model = Model(nn_jl)

    n = model.n
    m = model.m

    N = 51
    nomtraj = SMM.ALTRO(model, params.x0, params.xf, params.Q, params.R, params.Qf, N, dt)

    costfun(x, u) = norm(x[end] - params.xf)
    best_X, best_U, best_cost, best_nomcost = grid_search(model, task_d, zeros(model.n), nomtraj, params.x0, params.xf, costfun, dt)
    X = best_X
    U = best_U

    model_d = discretize_model(model, :rk4)

    errs = Vector{Float64}(undef, length(nomtraj.u))
    for i = 1:length(nomtraj.u)
        J = TO.PartedMatrix(task_d)
        jacobian!(J, task_d, nomtraj.x[i], nomtraj.u[i], dt)
        J_ = TrajectoryOptimization.PartedMatrix(task_d)
        jacobian!(J_, model_d, nomtraj.x[i], nomtraj.u[i], dt)
        errs[i] = norm(J - J_)
    end

    if viz
        N = size(X, 1)
        p = plot(X,xlabel="time step",title="State Trajectory",label=["x$i" for i in 1:n])
        display(p)
        #savefig(p, "stat_traj.png")
        robot = RigidBodyDynamics.parse_urdf(Float64, params.urdf)
        q = [X[k][1:convert(Int, model.n/2)] for k = 1:N]
        t = range(0,stop=N*dt,length=N)

        SMM.viz_traj(q, t, robot, urdf)
    end
    return best_cost, best_nomcost, errs, nomtraj.x, nomtraj.u, X, U
end

function run_jacobian_evaluation(mtype, taskname, loadpath, datapath, dt)
    params = TASKS[taskname]
    task = params.task
    task_d = discretize_model(task, :rk4)
    env = MechaModLearn.Env(task_d, randn(task_d.n), thetamask)

    qdim = round(Int, task_d.n/2)
    nn_jl = getfield(MechaModLearn, Symbol(mtype))(qdim, task_d.m, thetamask; hidden_sizes=params.hparams.hidden_sizes[mtype])
    nn_py = getfield(MechaModLearn.Torch, Symbol(mtype))(qdim, task_d.m, thetamask; hidden_sizes=params.hparams.hidden_sizes[mtype])
    MechaModLearn.Torch.load_state!(nn_py, loadpath)
    Flux.loadparams!(nn_jl, MechaModLearn.Torch.params(nn_py))

    JLD2.@load datapath valid_dataset

    errs_xx, errs_xu = compute_jacobian_norm(nn_jl, task, valid_dataset, dt)
    mean_err_xx = mean(errs_xx)
    std_err_xx = std(errs_xx)
    mean_err_xu = mean(errs_xu)
    std_err_xu = std(errs_xu)
    
    return mean_err_xx, std_err_xx, mean_err_xu, std_err_xu
end


function evaluate(args)
    @assert length(args) == 4
    mtype = args[1]
    task_name = args[2]
    eval_type = args[3]
    loadpath = args[4]
    
    dt = 0.05

    if occursin(":", loadpath)
        newloadpath = joinpath("results", dirname(joinpath(splitpath(loadpath)[5:end]...))) # TODO
        mkpath(newloadpath)
        run(`scp $loadpath $newloadpath`)

        loadpath = joinpath(newloadpath, basename(loadpath))
    end

    if eval_type == "jac"
        datapath = args[5]
        mean_err_xx, std_err_xx, mean_err_xu, std_err_xu = run_jacobian_evaluation(mtype, task_name, loadpath, datapath, dt)
        results = Dict("xx_mean" => mean_err_xx, "xx_std"=> std_err_xx, "xu_mean"=>mean_err_xu, "xu_std"=>std_err_xu)
        open(joinpath(dirname(loadpath), "jac.json"), "w") do io
            JSON.print(io, results)
        end
    elseif eval_type == "cost"
        cost, nomcost, jac_errs, nom_x, nom_u, X, U = run_cost_evaluation(mtype, task_name, loadpath, dt)
        results = Dict("cost"=>cost, "nomcost"=>nomcost,
                       "jac_err_mean"=>mean(jac_errs), "jac_err_std"=>std(jac_errs))
        println(JSON.json(results))
        open(joinpath(dirname(loadpath), "cost.json"), "w") do io
            JSON.print(io, results)
        end
    end
end
