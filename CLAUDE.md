# Swarm Simulation — O1 Predictive Model

Simulatore Python per dinamiche di swarm robotics: mobilità, RGG, 
stima locale di concentrazione spaziale (chi), diffusione SI.

## Struttura

- `core/arena.py` — boundary e dinamiche di movimento (random_walk_step, 
  aggregation_step, flocking_step). Boundary riflessivi di default, 
  toroidali per flocking.
- `core/swarm.py` — inizializzazione posizioni (uniform, gaussian)
- `core/graph.py` — costruzione RGG con cKDTree
- `metrics/chi.py` — chi_true (globale) e chi_hat (stima locale per robot)
- `simulation/runner.py` — SimRunner, classe principale che esegue un run
- `data/logger.py` — SimLogger, raccoglie dati per timestep
- `visualization/plots.py` — plot_chi_tracking, plot_SI_curve
- `visualization/snapshot.py` — plot_snapshot, make_gif
- `experiments/experiment.py` — Experiment, multi-run con seed progressivi
- `experiments/postprocess.py` — aggregazione su più run

## Convenzioni

- step_size = velocità fissa per timestep (non gaussiana), per coerenza con ARGoS
- mobility_mode: "random_walk" | "aggregation" | "flocking"
- Boundary riflessivi ovunque TRANNE flocking, che usa boundary toroidali 
  (distanze minimal-image in Arena._toroidal_delta)
- t_star: timestep di trigger SI, None disabilita la diffusione
- Diffusione SI deterministica (beta=1), non probabilistica

## Stato attuale / TODO

- Random walk: completo e validato
- Aggregation: funziona ma forma mini-cluster locali con R piccolo, 
  da rivedere R_sensing vs R_comm
- Flocking: completo — Reynolds con boundary toroidali, parametri 
  (R_sep/R_ali/R_coh, W_sep/W_ali/W_coh, flock_noise, dt, max_force, 
  min_speed) configurabili da SimRunner
- Manca ancora: organizzazione sistematica dei dati per confrontare 
  i tre mobility_mode
- chi_true assume box delimitato (std lineare), non corretto se un flock 
  attraversa il bordo periodico del toro — errore considerato raro/piccolo 
  per ora (il gruppo è coeso), da rivedere con statistica circolare se 
  emerge nei dati

## Note di design

- chi_hat si inverte da k_i (degree locale) assumendo modello saturante 
  calibrato con C_SAT = 0.911
- Ogni esperimento salva ogni singola run (non solo aggregato) in 
  results/{exp_name}/run_{i:03d}.csv — il postprocess è separato