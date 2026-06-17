# Swarm Simulation — O1 Predictive Model

Simulatore Python per dinamiche di swarm robotics: mobilità, RGG,
stima locale di concentrazione spaziale (chi), diffusione SI.

## Struttura

- `core/arena.py` — boundary e dinamiche di movimento (random_walk_step,
  aggregation_step, flocking_step). Boundary riflessivi di default,
  toroidali per flocking.
- `core/swarm.py` — inizializzazione posizioni (uniform, gaussian)
- `core/graph.py` — costruzione RGG con cKDTree; supporta boundary
  periodici (`L` param) per il caso flocking
- `metrics/chi.py` — chi_true (globale) e chi_hat (stima locale per robot)
- `simulations/runner.py` — SimRunner, classe principale che esegue un run
- `data/logger.py` — SimLogger, raccoglie dati per timestep
- `visualizations/plots.py` — plot_chi_tracking, plot_SI_curve
- `visualizations/snapshot.py` — plot_snapshot, make_gif
- `experiments/experiment.py` — Experiment, multi-run con seed progressivi
- `experiments/postprocess.py` — aggregazione e plot su più run
- `experiments/run_experiment.py` — script di esempio per un esperimento
- `main.py` — punto di ingresso con CONFIGS dict; cambia RUN per scegliere

## Convenzioni

- `step_size` = velocità fissa per timestep (non gaussiana), per coerenza
  con ARGoS
- `mobility_mode`: `"random_walk"` | `"aggregation"` | `"flocking"`
- Boundary riflessivi per random_walk e aggregation; toroidali per flocking
  (distanze minimal-image in `Arena._toroidal_delta`, RGG periodico via
  `cKDTree(boxsize=L)`)
- `t_star`: timestep di trigger SI, `None` disabilita la diffusione.
  Il seed avviene e viene loggato nello stesso timestep t_star, quindi
  `I(t_star) = 1/N > 0` nel CSV.
- Diffusione SI deterministica (beta=1), non probabilistica
- `Experiment` forwarda qualsiasi kwarg a SimRunner via `**runner_kwargs`
  — aggiungere nuovi parametri a SimRunner non richiede modifiche a Experiment

## Parametri per modalità

### aggregation
`agg_d_min=0.3, agg_R_personal=0.5, agg_W_attract=1.0, agg_W_repel=2.0, agg_noise=0.05`

Forza globale 1/d² verso tutti i robot (non limitata al RGG), con repulsione
a corto raggio per evitare collasso totale. Produce dinamica multi-scala:
raggruppamento locale rapido poi fusione inter-cluster più lenta.

### flocking
`R_sep=1.0, R_ali=3.0, R_coh=5.0, W_sep=2.0, W_ali=1.0, W_coh=1.0,`
`flock_noise=0.3, dt=0.1, max_force=0.5, min_speed=0.1`

Reynolds con tre zone (separazione, allineamento, coesione) e inerzia.
Tutto toroidale: forze, distanze e RGG usano la stessa metrica periodica.

## Stato

Tutti e tre i mobility mode sono **completi e verificati**:

| Mode | Boundary | RGG | Stato |
|------|----------|-----|-------|
| random_walk | riflessivo | Euclideo | ✅ completo |
| aggregation | riflessivo | Euclideo (solo per chi/SI, non per il movimento) | ✅ completo |
| flocking | toroidale | periodico (boxsize=L) | ✅ completo |

Check sistematici eseguiti: boundary [0,L] sempre rispettati, nessun NaN,
SI monotona, riproducibilità con stesso seed, edge case N=1/N=2/collasso
totale, pipeline runner → postprocess end-to-end per tutti i modi.

## Limitazioni note (design, non bug)

- **chi_true non periodica per flocking**: la formula usa std lineare delle
  posizioni, non statistica circolare. Se un flock attraversa il bordo del
  toro, chi_true risulta artificialmente alta. Evento raro con parametri
  attuali (flock coeso); da rivedere con von Mises se emerge nei dati.
- **chi_hat satura ad alta concentrazione**: robot con k_i = N−1 danno
  chi_hat = +inf, esclusi dal mean in to_summary. La media riflette solo
  il sotto-insieme non saturo; varianza alta è attesa in regime aggregato.
- **C_SAT = 0.911 calibrato su random_walk/statico**: chi_hat è
  qualitativamente corretto per aggregation/flocking (traccia la
  concentrazione) ma la calibrazione assoluta deriva da un regime diverso.
- **flocking_step usa un loop Python su N**: corretto, O(N²). Per N=100
  è ~5ms/step; per N >> 100 considerare vettorizzazione.

## Note di design

- `chi_hat` si inverte da k_i assumendo modello saturante:
  `E[k_i] = (N−1)·(1−exp(−η·(1+c·χ)/(N−1)))`, con C_SAT=0.911
- `eta = N·π·R²/L²` è calcolato da SimRunner e passato a chi_hat;
  per confronti a densità diverse, agire su R mantenendo N e L fissi
- Ogni run salva un CSV summary in `results/{exp_name}/run_{i:03d}.csv`;
  il postprocess legge tutti i run e produce `aggregate.csv` + plot
- `results/` è in .gitignore — i dati non sono versionati

## TODO aperti

- Sweep sistematica multi-run per confrontare i tre mobility_mode a
  diverse densità η (prossimo step naturale)
