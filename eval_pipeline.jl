using LinearAlgebra
using JSON
using ProgressMeter
using Flux
using TrajectoryOptimization
const TO = TrajectoryOptimization

using StructuredMechModels
const SMM = StructuredMechModels

include(joinpath(@__DIR__, "tasks.jl"))

function compute_jacobian_norm(nn, task, valid_dataset, dt)
    model = Model(nn)
    model_d = discretize_model(model, :rk4)
    task_d = discretize_model(task, :rk4)
    Z1 = TO.PartedMatrix(model_d)
    Z2 = TO.PartedMatrix(task_d)

    errs_xx = Vector{Float64}(undef, length(valid_dataset))
    errs_xu = Vector{Float64}(undef, length(valid_dataset))
    @showprogress 1 "Computing..." for i in 1:length(valid_dataset)
        x = valid_dataset.xs[:, i]
        u = valid_dataset.us[:, i]
        TO.jacobian!(Z1, model_d, x, u, dt)
        TO.jacobian!(Z2, task_d, x, u, dt)
        errs_xx[i] = norm(Z1.xx .- Z2.xx)
        errs_xu[i] = norm(Z1.xu .- Z2.xu)
    end
    return errs_xx, errs_xu
end

function compute_cost(X, U, Q, R, Qf)
    @assert size(X, 1) == (size(U, 1) + 1)
    cost = 0.
    for i in 1:length(U)
        cost += X[i]' * Q * X[i]
        cost += U[i]' * R * U[i]
    end
    cost += X[end]' * Qf * X[end]
    return cost
end

function grid_search(model, task, thetamask, nomtraj, x0, xf, costfun, dt)
    best_cost = Inf
    best_nomcost = Inf
    best_X = [zeros(model.n) for i in 1:100]
    best_U = [zeros(model.m) for i in 1:99]
    best_q_fac = 0.0
    best_r_fac = 0.0
    best_qf_fac = 0.0
    @info "Starting GridSearch..."
    for q_fac in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0, 10.0^4]
        for r_fac in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0, 10.0^4]
            for qf_fac in [1.0, 10.0, 100.0, 1000.0, 10000.0, 10.0^6]
                N = 51
                controller = SNN.TVLQRController(model, x0, xf, nomtraj, q_fac*Diagonal(I, model.n), r_fac*Diagonal(I, model.m), qf_fac*Diagonal(I, model.n), N, dt, 1.0)

                dt = controller.Δt

                env = SNN.Env(task, x0, thetamask)
                N = 100
                X, U = SNN.rollout(env, SNN.clip(controller, 200), N, dt)
                X = X'
                U = U'
                X = [X[i,:] for i = 1:size(X)[1]]
                U = [U[i,:] for i = 1:size(U)[1]]
                # cost = compute_cost(X, U, Q, R, Qf)
                # nomcost = compute_cost(controller.nom_x,controller.nom_u, Q, R, Qf)
                cost = costfun(X, U)
                nomcost = costfun(controller.nom_x, controller.nom_u)
                if cost < best_cost
                    @info "GS..." r_fac=r_fac q_fac=q_fac qf_fac=qf_fac
                    best_cost = cost
                    best_nomcost = nomcost
                    best_X = X
                    best_U = U
                    best_q_fac = q_fac
                    best_r_fac = r_fac
                    best_qf_fac = qf_fac
                end
            end
        end
    end
    @info "best..." best_cost=best_cost best_nomcost=best_nomcost best_q_fac=best_q_fac best_r_fac=best_r_fac best_qf_fac=best_qf_fac
    return best_X, best_U, best_cost, best_nomcost
end


function run_cost_evaluation(mtype, taskname, loadpath, dt; viz=false)
    params = TASK_REGISTRY[taskname]

    task_d = discretize_model(params.sys, :rk4)
    env = SNN.Env(task_d, randn(task_d.n), params.thetamask)
    qdim = round(Int, task_d.n/2)

    hidden_sizes = params.hparams.hidden_sizes
    nn_jl = getfield(SMM, Symbol(mtype))(qdim, task_d.m, params.thetamask; hidden_sizes=hidden_sizes)
    nn_py = getfield(SMM.Torch, Symbol(mtype))(qdim, task_d.m, params.thetamask; hidden_sizes=hidden_sizes)
    SNN.Torch.load_state!(nn_py, loadpath)
    Flux.loadparams!(nn_jl, SNN.Torch.params(nn_py))
    model = Model(nn_jl)

    n = model.n
    m = model.m

    N = 51
    nomtraj = SNN.ALTRO(model, params.x0, params.xf, params.Q, params.R, params.Qf, N, dt)

    costfun(x, u) = norm(x[end] - params.xf)
    best_X, best_U, best_cost, best_nomcost = grid_search(model, task_d, zeros(model.n), nomtraj, params.x0, parasm.xf, costfun, dt)
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

        SNN.viz_traj(q, t, robot, urdf)
    end
    return best_cost, best_nomcost, errs, nomtraj.x, nomtraj.u, X, U
end

function run_jacobian_evaluation(mtype, taskname, loadpath, datapath, dt)
    params = TASKS[taskname]
    task = params.task
    task_d = discretize_model(task, :rk4)
    env = MechaModLearn.Env(task_d, randn(task_d.n), thetamask)

    qdim = round(Int, task_d.n/2)
    nn_jl = getfield(MechaModLearn, Symbol(mtype))(qdim, task_d.m, thetamask; hidden_sizes=params.hparams.hidden_sizes)
    nn_py = getfield(MechaModLearn.Torch, Symbol(mtype))(qdim, task_d.m, thetamask; hidden_sizes=params.hparams.hidden_sizes)
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
