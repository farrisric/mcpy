# Trial energies are LBFGS-relaxed, not single-point

`BaseCalculator.get_potential_energy` runs an LBFGS geometry relaxation (up to `steps` / `fmax`) on each trial configuration and returns the *relaxed* energy, which is what the acceptance criterion sees. Standard Metropolis Monte Carlo evaluates the single-point energy of the configuration as proposed; mcpy relaxes first by default.

We chose this because the target use case is expensive ML potentials (MACE, Alchemi) on nanoparticles, where relaxing each trial lets the sampler find and accept genuinely stable structures rather than rejecting almost every move on raw, unrelaxed overlaps. The trade-off is that the sampled distribution is then over *relaxed basins*, not the literal configurational ensemble — so acceptance ratios and any thermodynamic quantities should be read as basin-level, not strict NVT/μVT sampling.

This is configurable, not hard-coded: passing `steps=0` recovers single-point Metropolis energies for a study that needs true configurational sampling. `fmax` and `steps` trade relaxation depth against cost per step.
