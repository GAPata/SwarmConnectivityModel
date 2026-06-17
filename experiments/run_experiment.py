from experiments.experiment import Experiment

experiments = [
    Experiment(
        exp_name="test_N100_R1.0_v0.3_uniform_tstar50",
        n_runs=5,
        N=100, L=20.0, R=1.0, step_size=0.3, T_steps=200,
        init_mode="uniform", t_star=50,
        mobility_mode="random_walk",
    ),
    Experiment(
        exp_name="test_N100_R1.0_v0.3_uniform_tstar100",
        n_runs=5,
        N=100, L=20.0, R=1.0, step_size=0.3, T_steps=200,
        init_mode="uniform", t_star=100,
        mobility_mode="aggregation",
    ),
    Experiment(
        exp_name="test_N100_R2.0_v1.0_sigma3.0_tstar50",
        n_runs=5,
        N=100, L=20.0, R=2.0, step_size=1.0, T_steps=200,
        init_mode="gaussian", sigma_init=3.0, t_star=50,
        mobility_mode="flocking",
    ),
]

for exp in experiments:
    print("-" * 60)
    exp.run()
print("-" * 60)
print("All experiments done.")
