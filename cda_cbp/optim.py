"""The authors' AdamGnT, with its per-step scalar allocations removed.

`lop.utils.AdamGnT` (imported unchanged from the submodule) allocates two
scalar tensors - `beta1_t` and `beta2_t` - for every parameter on every step.
That is invisible at the model sizes this package runs on CPU, but on a GPU
learner it adds an allocation and a device transfer per parameter per step
(#16, #24). The optimiser maths is the authors' and must not change: this
subclass copies `AdamGnT.step` and caches the two beta tensors per parameter
group instead. `tests/test_adam_gnt_optim.py` compares it with the original
step by step, and `tests/test_lop_reference.py` keeps the CBP reference run on
it.

The `lop` files themselves stay untouched (`tests/test_lop.py` checks bytes).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from lop.utils.AdamGnT import AdamGnT


class CBPAdamGnT(AdamGnT):
    """`AdamGnT` with one `beta1_t` / `beta2_t` pair per param group.

    The cache is keyed by the group object and device, and stores the beta
    values it was built from, so a group whose `betas` are changed later gets a
    fresh pair rather than a stale scalar.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cbp_beta_tensors: Dict[
            Tuple[int, torch.device], Tuple[Any, Any, torch.Tensor, torch.Tensor]
        ] = {}

    def _beta_tensors(
        self, group: Dict[str, Any], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (id(group), device)
        beta1, beta2 = group["betas"]
        cached = self._cbp_beta_tensors.get(key)
        if cached is not None and cached[0] == beta1 and cached[1] == beta2:
            return cached[2], cached[3]
        beta1_t = torch.tensor(beta1, device=device)
        beta2_t = torch.tensor(beta2, device=device)
        self._cbp_beta_tensors[key] = (beta1, beta2, beta1_t, beta2_t)
        return beta1_t, beta2_t

    def step(self, closure: Optional[Any] = None):
        """One AdamGnT step, with the beta scalars cached per group.

        This is the authors' `AdamGnT.step` with only the two `torch.tensor`
        constructions moved out of the per-parameter loop. Every arithmetic
        operation, including the per-weight `step`, is unchanged.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.zeros_like(p.data)
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                beta1_t, beta2_t = self._beta_tensors(group, p.data.device)
                bias_correction1 = 1 - torch.pow(beta1_t, state["step"])
                bias_correction2 = 1 - torch.pow(beta2_t, state["step"])
                step_size = group["lr"] * (bias_correction2.sqrt() / bias_correction1)

                update = exp_avg / denom
                update.mul_(step_size)
                p.data.add_(-update)

        return loss
