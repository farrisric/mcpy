from mace.calculators import MACECalculator
from ase.optimize import LBFGS, FIRE
from typing import Optional, Union

from .base_calculator import BaseCalculator


_OPTIMIZERS = {'lbfgs': LBFGS, 'fire': FIRE}


class MACE_F_Calculator(BaseCalculator):
    def __init__(self, model_paths: Union[str, MACECalculator],
                 steps: int, fmax: float,
                 device: str = 'cpu', cueq: bool = False,
                 optimizer: str = 'lbfgs',
                 head: Optional[str] = None,
                 default_dtype: str = '') -> None:
        """
        Initialize the Calculator with the given model path and device.

        :param model_paths: Path to the model file, or an initialized
                            MACECalculator instance to reuse directly.
        :param device: Device to load the model on ('cpu' or 'cuda').
        :param optimizer: ASE optimizer to use: 'lbfgs' (default) or 'fire'.
                          Use 'fire' for apples-to-apples comparison with
                          AlchemiFCalculator (which also uses FIRE).
        :param head: Head of a multihead model, by name. ``None`` keeps
                     MACECalculator's default (the last head).
        :param default_dtype: 'float32' or 'float64'; '' keeps the
                              MACECalculator default.

        Relaxation, step counters and the step-cap warning come from
        :class:`BaseCalculator`.
        """
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f'optimizer must be one of {list(_OPTIMIZERS)}, got {optimizer!r}')

        if isinstance(model_paths, MACECalculator):
            if head is not None or default_dtype:
                raise ValueError(
                    'head= and default_dtype= cannot be applied to a '
                    'pre-built MACECalculator; pass them when you build it, '
                    'or pass the model path here.'
                )
            calculator = model_paths
        else:
            # head goes through MACECalculator's **kwargs; only pass it when
            # set, so ASE does not record a head=None parameter.
            extra = {} if head is None else {'head': head}
            calculator = MACECalculator(
                model_paths=model_paths,
                device=device,
                enable_cueq=cueq,
                default_dtype=default_dtype,
                **extra,
            )
        super().__init__(calculator, steps, fmax)
        self._optimizer_cls = _OPTIMIZERS[optimizer]
        self.optimizer_name = optimizer
