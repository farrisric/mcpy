# Batched RE-GCMC scaling (AlchemiFCalculator, FIRE relax)

Whole batch (chunk_size=None), fmax=0.05, relax_steps=500, gcmc_steps=10 (warmup 2). Card: 32 GB RTX 5090.

## Peak GPU memory, nvidia-smi (MB)

| atoms \\ replicas | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 140 | 1282 | 1982 | 3686 | 7868 |
| 338 | 2100 | 5210 | 12444 | 21810 |
| 664 | 2796 | 7806 | 15168 | 29021 |

## Peak torch allocated (MB)

| atoms \\ replicas | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 140 | 327.7 | 559 | 1013.3 | 1934.7 |
| 338 | 710.2 | 1335.8 | 2552.9 | 4955 |
| 664 | 1369.9 | 2650.8 | 5166.4 | 10248.1 |

## Mean GPU utilization (%)

| atoms \\ replicas | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 140 | 17.5 | 21.9 | 30.3 | 45.2 |
| 338 | 24.7 | 36.3 | 58.2 | 73.3 |
| 664 | 42.1 | 57.1 | 75.6 | 83.5 |

## Throughput (s / GCMC step)

| atoms \\ replicas | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 140 | 1.638 | 2.706 | 2.189 | 2.971 |
| 338 | 1.45 | 2.658 | 3.145 | 4.594 |
| 664 | 1.39 | 1.854 | 3.68 | 9.311 |

