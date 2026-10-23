# Metrics

Emitted per trainable module by `CBPLearnerMixin` when the `cbp` extension is enabled, with or
without `metrics_only`, through the same `self.metrics.log_dict(..., key=module_id, window=1)` path
gym-CDA's MoE term and cda-jepa's JEPA terms use. Absent entirely when the extension is not enabled. Full context
in [continual_backprop.md](continual_backprop.md) §4.1.

The first three are the **three correlates of loss of plasticity** both papers track. The Nature
paper's claim for continual backprop is precisely that it is the only method that keeps all three
healthy at once, so they are read together rather than individually.

| Metric | Reduction across layers | What a bad value looks like |
|---|---|---|
| `cbp_dead_unit_frac` | max | Rising. Units whose mean \|activation\| has fallen below the threshold and stopped contributing |
| `cbp_mean_weight_magnitude` | mean | Rising. Weight growth is the correlate L2 addresses and CBP also suppresses |
| `cbp_batch_effective_rank` | mean | **Confounded — see below.** Units becoming redundant, but measured on the training minibatch |
| `cbp_saturated_unit_frac` | max | Rising. Specific to tanh, which is this project's default: past \|h\| > 0.9 the local gradient vanishes |
| `cbp_utility_min` / `_median` | min / mean | A widening gap means capacity is concentrating in fewer units |
| `cbp_mature_unit_frac` | mean | **Pinned at 0.0 means the mechanism cannot fire at all** — see below |
| `cbp_replacements` | **sum** | **Flat at 0 means it never fired** |

`cbp_replacements` is summed rather than averaged because it counts events in the network rather
than describing a layer; `cbp_dead_unit_frac` and `cbp_saturated_unit_frac` take the max and
`cbp_utility_min` the min, because one collapsing layer is the thing worth seeing and an average
over healthy neighbours would hide it. The rest are averaged. Reduced to one value per module
rather than emitted per layer, which would be 80 series for a 16-layer `moe_transformer`.

**The two that decide whether any of the others mean anything** are `cbp_replacements` and
`cbp_mature_unit_frac`. A continual backprop that never fires — which is the *expected* outcome at
the shipped replacement rate over a short run, for the cadence reason in
[continual_backprop.md](continual_backprop.md) §3.6 — logs exactly what a working one logs. Read those two first.

`cbp_batch_effective_rank` runs an SVD and is therefore computed every `metrics_every_n_updates`
optimiser steps rather than every one; the same counter gates keeping an activation sample in the
forward hook at all.

### The rank metric is confounded, and was renamed to say so

The rank of an activation matrix depends on the inputs as much as on the network. This one is
measured on whatever minibatch training produced, and the policy chooses its own inputs — so under
S1-3, where passivity is the joint optimum, a converging agent visits a narrower set of book states
and this falls **with the network unchanged**.

Measured ([16](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction/blob/master/docs/16_verification_log.md) §16.16), over ~18,000 optimiser steps: this metric fell
**24.6%** while the same network's rank on a fixed corpus moved by **+2.5% to −1.9%**. Read on its
own it is a textbook plasticity collapse; it is the agent's behaviour.

It is kept because "what the network is doing on the data it is training on" is a real thing to
want, and it is named for the batch it is measured on rather than for what it looks like it
measures. **For a statement about the network, use `train/probe/rank.py`** across checkpoints
([23](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction/blob/master/docs/23_probe_harness.md) §10), where the corpus is fixed while the encoder varies.

The other two correlates are robust to this — weights do not depend on the batch, and a unit dead
on one input distribution is generally dead on others — but both read zero on the shipped network
for their own reasons: `cbp_dead_unit_frac` is ReLU-shaped and a tanh unit does not die toward
zero, and `cbp_saturated_unit_frac` uses the per-unit batch mean, which is stricter than the
papers' per-output definition.
