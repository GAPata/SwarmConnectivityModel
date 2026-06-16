from experiments.experiment import Experiment

exp = Experiment(
    exp_name   = "N100_R0.8_v0.3_sigma2.0_tstar50",
    n_runs     = 50,
    N          = 100,
    L          = 20.0,
    R          = 0.8,
    step_size  = 0.3,
    T_steps    = 800,
    init_mode  = "uniform",
    sigma_init = 2.0,
    t_star     = 50,
)

exp.run()
