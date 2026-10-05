# AI4S Thin-Film MLP Project

Student ID: 2023270177
Target wavelength: 600 nm
seed: 270177
design_seed: 270178

## Reproduction
The project uses an Air/H/L/H/L/Glass four-layer system with nH=2.30,
nL=1.45, ns=1.52. Thicknesses are sampled uniformly from 40–180 nm.
The TMM spectrum is calculated from 400–800 nm at 10 nm intervals.

Dataset: 5000 structures, split 4000/500/500 after a seed-controlled permutation.
MLP: 4-128-128-64-41, ReLU activations, Adam, learning rate 1e-3,
batch size 128, MSE loss. Training-size experiments use 500/1000/2000/4000
samples from the same training pool.

For design, design_seed=270178 generates 10000 new candidates.
The trained MLP ranks candidates at 600 nm; the Top 10 are
recomputed using TMM and the final Top 5 are selected from the verified values.

All numerical results in the report are generated from this same workflow.
