const TASK_REGISTRY = Dict(
    "furuta"=>(sys=SMM.Dynamics.furuta_model, thetamask=[1, 1, 0, 0],
               urdf=SMM.Dynamics.urdf_furuta,
               x0=[0.0, 0.0, 0.0, 0.0],
               xf=[0.0, π, 0.0, 0.0],
               Q=(1e-3)*Diagonal(I, 4),
               Qf=(1e3)*Diagonal(I, 4),
               R=(1e-3)*Diagonal(I, 1),
               hparams=(gradnorm=100., maxu=100., stddev=30.0, patience=500, lr=1e-3, noisestd=0.001, 
                        hidden_sizes=Dict("ControlAffine"=>(32, 32, 32) , "Naive"=>(128, 128, 128)))),
    "cartpole"=>(sys=TO.Dynamics.cartpole_urdf, thetamask=[0, 1, 0, 0],
                 urdf=TO.Dynamics.urdf_cartpole,
                 x0=[0.0, -π, 0.0, 0.0],
                 xf=[0.0, 0.0, 0.0, 0.0],
                 Q=(1e-2)*Diagonal(I, 4),
                 Qf=(1e3)*Diagonal(I, 4),
                 R=(10.0)*Diagonal(I, 1),
                 hparams=(gradnorm=100., maxu=100., stddev=30.0, patience=500, lr=1e-3, noisestd=0.001, 
                          hidden_sizes=Dict("ControlAffine"=>(32, 32, 32) , "Naive"=>(128, 128, 128)))),
    "acrobot"=>(sys=TO.Dynamics.acrobot_model, thetamask=[1, 1, 0, 0],
                urdf=TO.Dynamics.urdf_doublependulum,
                x0=[0.0, 0.0, 0.0, 0.0],
                xf=[π, 0.0, 0.0, 0.0],
                Q=(1e-3)*Diagonal(I, 4),
                Qf=(1e3)*Diagonal(I, 4),
                R=(1e-3)*Diagonal(I, 1),
                hparams=(gradnorm=100., maxu=100., stddev=30.0, patience=500, lr=1e-3, noisestd=0.001, 
                         hidden_sizes=Dict("ControlAffine"=>(32, 32, 32) , "Naive"=>(128, 128, 128)))),
    "doublecartpole"=>(sys=SMM.Dynamics.doublecartpole_model, thetamask=[0, 1, 1, 0, 0, 0],
                       urdf=SMM.Dynamics.urdf_doublecartpole,
                       x0=[0.0, 0.0, 0.0, 0.0],
                       xf=[π, 0.0, 0.0, 0.0],
                       Q=(1e-2)*Diagonal(I, 6),
                       Qf=(1e3)*Diagonal(I, 6),
                       R=(1e-2)*Diagonal(I, 1),
                       hparams=(gradnorm=1000., maxu=100., stddev=30.0, patience=2000, lr=1e-3, noisestd=0.00001,
                                hidden_sizes=Dict("ControlAffine"=>(64, 64, 64, 64), "Naive"=>(256, 256, 256, 256)))),
)
