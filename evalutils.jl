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
                controller = SMM.TVLQRController(model, x0, xf, nomtraj, q_fac*Diagonal(I, model.n), r_fac*Diagonal(I, model.m), qf_fac*Diagonal(I, model.n), N, dt, 1.0)

                dt = controller.Δt

                env = SMM.Env(task, x0, thetamask)
                N = 100
                X, U = SMM.rollout(env, SMM.clip(controller, 200), N, dt)
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
