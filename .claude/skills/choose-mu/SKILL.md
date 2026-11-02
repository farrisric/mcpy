---
name: choose-mu
description: Use when setting the chemical potential mu for a GCMC run - deriving reference mu from gas or bulk, choosing a delta_mu range, converting to pressure, or when a mu value from a paper or another potential is being reused.
---

# Choose the chemical potential

Rule one: derive the reference mu with the production calculator at runtime.
Each checkpoint gives a different reference (E(O2)/2 differs per model), so a number
copied from a paper or another potential is wrong.

## Steps

1. Reference mu from `mcpy.utils.reference_mu`:
   - gas reservoir: `derive_mu_gas(calculator, species='O2')` = per-atom relaxed molecule
     energy. Homonuclear only (O2, H2, N2); CO raises `ValueError` - for a heteronuclear
     molecule derive the molecular mu per `docs/molecular_adsorbates.rst`.
   - metal reservoir: `derive_mu_bulk(calculator, symbol, crystalstructure, a)` - EOS fit,
     cubic structures only.
2. Run at `mu = mu_ref + delta_mu`. Scan `delta_mu` (e.g. 0 to -1 eV) to map coverage;
   `mcpy.utils.phase_diagram.from_mu_to_press` converts to pressure for the paper.
3. Sanity-check the scale: if the adsorption energy per atom is only ~0.2 eV below the
   reference, `delta_mu = -0.5` giving zero insertions is correct physics, not a bug.
4. For a mu-ladder replica exchange, spacing is NOT uniform - use the
   `replica-exchange-setup` skill.

Done when: mu_ref comes from the running calculator and the delta_mu range brackets the
coverage transition you care about.

## Traps

- Hardcoded mu_O = -4.91: correct only for one checkpoint (mace-small-density); wrong for
  medium-mpa-0 and mh-1. Always re-derive.
- Changing `species_radii` changes the free volume, which shifts the effective mu by
  ~0.01-0.04 eV; re-calibrated radii mean re-checked transitions.
