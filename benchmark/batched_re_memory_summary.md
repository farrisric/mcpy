# Batched RE memory sweep

## Noise floor per size (eV)
- 256 atoms: 1.221e-04
- 1024 atoms: 1.221e-03
- 2048 atoms: 8.789e-03
- 3584 atoms: 5.859e-03

## Min-safe max_neighbors per size
- 256 atoms: min-safe = 16
- 1024 atoms: min-safe = 16
- 2048 atoms: min-safe = 16
- 3584 atoms: min-safe = 16

## energy_only saving (within noise?)
- 256 atoms x1: 91.0% of baseline, max|dE|=1.83e-04 eV, within_noise=False
- 256 atoms x2: 90.1% of baseline, max|dE|=1.22e-04 eV, within_noise=True
- 256 atoms x4: 89.4% of baseline, max|dE|=1.22e-04 eV, within_noise=True
- 256 atoms x8: 89.2% of baseline, max|dE|=3.05e-04 eV, within_noise=True
- 1024 atoms x1: 88.6% of baseline, max|dE|=4.88e-04 eV, within_noise=True
- 1024 atoms x2: 88.3% of baseline, max|dE|=4.88e-04 eV, within_noise=True
- 1024 atoms x4: 88.2% of baseline, max|dE|=3.66e-03 eV, within_noise=False
- 1024 atoms x8: 88.1% of baseline, max|dE|=5.37e-03 eV, within_noise=False
- 2048 atoms x1: 88.1% of baseline, max|dE|=5.37e-03 eV, within_noise=True
- 2048 atoms x2: 87.9% of baseline, max|dE|=3.42e-03 eV, within_noise=True
- 2048 atoms x4: 87.9% of baseline, max|dE|=1.71e-02 eV, within_noise=False
- 3584 atoms x1: 87.8% of baseline, max|dE|=1.17e-02 eV, within_noise=False
- 3584 atoms x2: 87.7% of baseline, max|dE|=1.95e-02 eV, within_noise=False
