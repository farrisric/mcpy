Cluster install: IQTC portal (mcpy + Alchemi on Ubuntu GPU nodes)
=================================================================

This recipe sets up an environment that runs ``mcpy`` GCMC simulations with
the NVIDIA Alchemi GPU stack (``AlchemiCalculator`` /
``AlchemiFCalculator``) on the IQTC portal cluster's **iqtc13.q** partition
(RTX 4090 nodes ``merry04`` / ``merry05``).

It is written for that cluster's actual layout — the login node and the
compute nodes have different operating systems, so a naive ``conda
activate`` in a SLURM script does not work.

The same approach generalises to any cluster where the login and compute
nodes diverge in OS, drivers, or mounts.


Cluster facts you need to know
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 22 8 16 8 10

   * - Node
     - Role
     - OS
     - glibc
     - ``/aplic`` conda
     - ``module``
     - Internet
   * - ``portal03``
     - login
     - Debian 10
     - 2.28
     - yes (``/aplic/anaconda/2020.02``)
     - yes
     - yes
   * - ``merry04``, ``merry05``
     - iqtc13.q GPU
     - Ubuntu 24.04
     - 2.39
     - **NO** (path absent)
     - **NO**
     - **NO**

Consequences:

- Build and install on the **login node** (only place with internet + conda).
- Run on the **compute nodes** using the env's ``python`` directly — **do
  not** call ``conda activate`` from inside the SLURM script (the path
  doesn't exist on Ubuntu nodes and the script will fail silently).
- Pre-download any model checkpoints on the login node; compute nodes can't
  reach the internet.
- ``merry04/05`` driver is **570.x → CUDA 12.8**. Use PyTorch built for
  ``cu128`` (``cu130`` wheels will see ``torch.cuda.is_available() ==
  False``).

``/home`` is NFS-mounted on every node, so a conda env created on
``portal03`` is reachable from ``merry04/05``.


One-time setup on ``portal03``
------------------------------

Workspace prep
~~~~~~~~~~~~~~

NFS-safe scratch for pip. The default ``/tmp`` is only ~1 GB and ``/home``
is NFS, which trips pip with "Directory not empty" errors during installs.

.. code-block:: bash

   mkdir -p /dev/shm/$USER-pip
   export TMPDIR=/dev/shm/$USER-pip
   export TEMP=$TMPDIR TMP=$TMPDIR

Create a fresh conda env
~~~~~~~~~~~~~~~~~~~~~~~~

Always use a **fresh env**. Mixing ``nvalchemi`` / MACE / cuequivariance
into an existing env almost always breaks because of pinned CUDA-specific
dependencies.

.. code-block:: bash

   source /aplic/anaconda/2020.02/etc/profile.d/conda.sh
   conda create -n alchemi128 python=3.11 pip -y

Install PyTorch with CUDA 12.8 wheels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ENV_PY=$HOME/.conda/envs/alchemi128/bin/python
   PYTHONNOUSERSITE=1 $ENV_PY -m pip install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cu128

The ``+cu128`` build matches the driver on ``merry04/05``. Don't take
whatever ``pip install torch`` gives you by default — that's ``+cu130``
and will run on CPU only.

Install nvalchemi + MACE + mcpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   PYTHONNOUSERSITE=1 $ENV_PY -m pip install --no-cache-dir --prefer-binary \
       'nvalchemi-toolkit[mace]'

   # mcpy from your clone (editable)
   PYTHONNOUSERSITE=1 $ENV_PY -m pip install --no-cache-dir --prefer-binary \
       -e $HOME/bin/mcpy

Resulting key versions (as of this writing):

.. code-block:: text

   torch                            2.11.0+cu128
   nvalchemi-toolkit                0.1.0
   nvalchemi-toolkit-ops            0.3.1
   cuequivariance-torch             0.10.0
   cuequivariance-ops-torch-cu12    0.10.0     # matches your CUDA major
   mace-torch                       0.3.15
   e3nn                             0.4.4
   ase                              3.28.0
   numpy                            2.4.6

Pre-download the MACE-MP checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute nodes have no internet, so cache the model on the login node. The
cache directory (``~/.cache/mace/``) is on NFS, so ``merry04/05`` will see
it.

.. code-block:: bash

   PYTHONNOUSERSITE=1 $ENV_PY -c "
   from mace.calculators.foundations_models import download_mace_mp_checkpoint
   print(download_mace_mp_checkpoint('medium-mpa-0'))
   "
   # → /home/$USER/.cache/mace/macempa0mediummodel  (≈ 80 MB)

Verify on a compute node
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ssh merry04 'PYTHONNOUSERSITE=1 $HOME/.conda/envs/alchemi128/bin/python -c "
   import torch
   print(\"torch:\", torch.__version__, \"cu:\", torch.version.cuda)
   print(\"cuda available:\", torch.cuda.is_available())
   print(\"cuda devices:\", torch.cuda.device_count())
   from mcpy.calculators import AlchemiCalculator, AlchemiFCalculator
   print(\"mcpy alchemi calculators: ok\")
   "'

Expected:

.. code-block:: text

   torch: 2.11.0+cu128 cu: 12.8
   cuda available: True
   cuda devices: 3
   mcpy alchemi calculators: ok

If ``cuda available: False``, the driver is older than CUDA 12.8 —
reinstall torch from the matching ``cu12X`` wheel index (e.g. ``cu124`` for
driver 550).


SLURM submit script
-------------------

Save as ``submit_gcmc_alchemi.sh``:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=gcmc_alchemi
   #SBATCH --partition=iqtc13.q
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=4
   #SBATCH --time=24:00:00
   #SBATCH --output=%x_%j.out
   #SBATCH --error=%x_%j.err

   set -eo pipefail

   PYTHON=$HOME/.conda/envs/alchemi128/bin/python
   SCRIPT=$HOME/bin/mcpy/examples/gcmc_nano_alchemi.py
   RESULTS_DIR=$SLURM_SUBMIT_DIR/results/job_${SLURM_JOB_ID:-local}

   export PYTHONNOUSERSITE=1
   export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
   export TMPDIR=/tmp/$USER
   mkdir -p "$TMPDIR" "$RESULTS_DIR"

   echo "=== node=$(hostname)  $(date) ==="
   nvidia-smi -L
   "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cu_avail', torch.cuda.is_available())"

   "$PYTHON" "$SCRIPT" \
       --checkpoint medium-mpa-0 \
       --device cuda \
       --no-compile \
       --T 500.0 \
       --steps 1000 \
       --delta-mu-O -0.5 \
       --fmax 0.05 \
       --relax-steps 500 \
       --seed 42 \
       --outdir "$RESULTS_DIR" \
       --write-interval 1

Submit with ``sbatch submit_gcmc_alchemi.sh``.

Key SLURM-script rules:

- **No** ``conda activate``. Use ``$HOME/.conda/envs/alchemi128/bin/python``
  directly.
- **No** ``module load``. Ubuntu compute nodes don't have environment
  modules.
- **Always export** ``PYTHONNOUSERSITE=1`` to ignore
  ``~/.local/lib/python3.11/site-packages``, which on many user homes
  contains stale packages that shadow the env.
- ``TMPDIR=/tmp/$USER`` — local disk on the compute node, not NFS.
- **Always pass** ``--no-compile`` to the simulation script. GCMC has a
  dynamic atom count (insertions/deletions change system size), so
  ``torch.compile`` would recompile on every new size and OOMs during the
  guard-building phase. ``compile_model=False`` is required for GCMC.


Common failure modes
--------------------

.. list-table::
   :header-rows: 1
   :widths: 32 36 32

   * - Symptom
     - Cause
     - Fix
   * - Job exits 127 with empty ``.out``/``.err``
     - Script tried ``conda activate`` on a node where ``/aplic/anaconda/``
       doesn't exist; ``set -e`` killed it.
     - Use the env's ``python`` binary directly (no activation).
   * - ``python: /lib64/libc.so.6: version 'GLIBC_2.28' not found``
     - Job landed on an old node (e.g. ``g9noder25o``) with glibc < 2.28.
     - Restrict the partition to modern nodes (``iqtc13.q`` is safe).
   * - ``torch.cuda.is_available() == False`` despite GPU present, with
       ``nvidia-smi`` driver "too old"
     - PyTorch built for newer CUDA than the driver supports.
     - Reinstall ``torch`` from the matching ``cu12X`` index.
   * - ``urllib.error.URLError: Name or service not known`` while loading
       the MACE checkpoint
     - Compute node has no internet.
     - Pre-download the model on the login node (see above).
   * - ``cuequivariance_ops_torch_cu11`` errors at runtime, or ``e3nn``
       version too old
     - ``~/.local`` user-site shadows the env.
     - Always export ``PYTHONNOUSERSITE=1``.
   * - ``pip install ... Directory not empty``
     - ``TMPDIR`` is on NFS.
     - ``export TMPDIR=/dev/shm/$USER-pip``.
   * - ``InternalTorchDynamoError: MemoryError: std::bad_alloc`` or
       ``Warp CUDA error 2: out of memory`` at startup
     - ``torch.compile`` tries to JIT-compile the model; the guard-building
       pass OOMs. GCMC also invalidates the cache on every insert/delete
       because atom count changes.
     - Pass ``--no-compile`` to the script (sets ``compile_model=False``).


Reproducing this exact env later
--------------------------------

.. code-block:: bash

   source /aplic/anaconda/2020.02/etc/profile.d/conda.sh
   conda env remove -n alchemi128 -y
   # then re-run the "One-time setup on portal03" section

The whole install takes ~5 minutes on a good connection and uses ~7 GB.
