# Settings

Every key the `cbp` extension accepts, with its default. gym-CDA fills in defaults and rejects a
key not listed here when the `TrainConfig` is built. The source of truth is
[`cda_cbp/config.py`](../cda_cbp/config.py) (`DEFAULTS`); `tests/test_settings_doc.py` checks this table against it.

```json
"extensions_enabled": ["cbp"],
"extension_settings": {"cbp": {"replacement_rate": 1e-4, "scope": "feedforward"}}
```

Listing `"cbp"` turns Continual Backprop on; there is no separate `enabled` key. Each key was
`cbp_<key>` in gym-CDA's `train_config.json` before CBP moved here, with the same default.

| Key | Default | Meaning |
|---|---|---|
| `metrics_only` | `false` | Measure the correlates of plasticity loss without replacing anything |
| `replacement_rate` | `0.0001` | Units replaced per mature unit per optimiser step, through an accumulator |
| `maturity_threshold` | `100` | Optimiser steps a unit must age before it can be replaced |
| `utility_decay` | `0.99` | Decay of the running utility and mean-activation averages, in [0, 1) |
| `utility` | `"overall"` | `contribution`, `mean_corrected` or `overall` |
| `scope` | `"feedforward"` | `feedforward`, or `recurrent` to add an LSTM core's top layer |
| `fire_on` | `"adam_step"` | `adam_step` (after every optimiser step) or `iteration` |
| `reset_optimizer_state` | `true` | Clear Adam's moments for a replaced unit's weights |
| `compensate_mean` | `false` | Move a replaced unit's mean contribution into the next layer's bias |
| `dead_unit_threshold` | `0.01` | Mean \|activation\| below which a unit counts as dead (metric only) |
| `metrics_every_n_updates` | `10` | How often, in updates, the effective-rank metric is computed |

## Rules

- **One learner when replacing.** `num_learners > 1` is refused unless `metrics_only` is true: each
  learner picks the unit to replace from its own minibatch, outside the gradient, so the replicas would
  drift apart (gym-CDA R-12, issue #182).
- **Not together with gym-CDA's own CBP.** While a gym-CDA version still has `cbp_enabled`, setting it
  as well as enabling this extension is refused: CBP would be applied twice.
- **Fixed for a run.** gym-CDA fingerprints every resolved setting as `ext.cbp.<key>`, so changing one on
  a resume raises, as does enabling the extension on a resume. Changing a setting means a fresh run.

## Notes carried over from gym-CDA's `train_config.json`

These explain the choices behind the defaults. They are kept verbatim, so they use the old `cbp_` names
and gym-CDA's config vocabulary (`UNRESTORABLE_CONFIG_KEYS` and the like); the rules above are how
the same guarantees hold now.

### What it is

Continual Backprop (Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood & Sutton, Nature 2024; Dohare, Sutton & Mahmood, arXiv 2108.06325). Backprop plus a generate-and-test process that continually reinitialises the lowest-utility hidden units: incoming weights resampled from the layer's initial distribution, outgoing weights set to zero so the fresh unit reaches the output not at all. It is an update rule, not an architecture, so it composes over whichever Learner the encoder selected rather than occupying the encoder registry's single learner slot - see gym-CDA docs/25 section 2.5. Off by default, and when off the resolved learner class is identical to the one a run would have had before this group existed.

### Resumes

cbp_enabled and cbp_metrics_only are UNRESTORABLE_CONFIG_KEYS: changing either in the same pass as is_restore is a hard error, and so is changing any of the tuning knobs below while continual backprop is running on either side (UNRESTORABLE_WHEN_ACTIVE_KEYS - a knob edited while the mechanism is off in both the checkpoint and the config is allowed, because it describes something that was not going to happen either way). Not for the reason encoder_spec is - no tensor shape depends on these, and the weights would restore perfectly well - but because they are consumed when the Learner class is chosen and the optimiser is built, both of which happen before a restore, after which Algorithm.from_checkpoint rebuilds from the checkpoint's own config and discards them. Turning continual backprop on alongside a resume therefore did nothing at all, and looked identical to a run that honoured it. Changing any of these means a fresh run, not a resume. The per-unit utility and ages ARE checkpoint state and are restored with the learner; a checkpoint written before this group existed restores cleanly and starts every unit at age 0.

### Cadence

cbp_replacement_rate and cbp_maturity_threshold are both counted in OPTIMISER STEPS - not environment steps, and not training iterations. This repo takes 4 Adam steps per iteration at its defaults (num_epochs 4 over one full batch, minibatch_size null) against the 320 of the papers' Continual PPO (10 epochs x 32 minibatches), so the papers' own PPO values cannot be transplanted: their maturity threshold of 1e4 would be 2,500 iterations here, against a default run of 16, and no unit would ever become mature. cbp_maturity_threshold is therefore the arXiv CBP default of 100 rather than the Nature PPO value. Watch cbp_replacements and cbp_mature_unit_frac: a continual backprop that never fires looks exactly like one that is working, and at low replacement rates over a short run not firing is the expected outcome.

### Metrics only

cbp_metrics_only computes the utility and logs the three correlates of loss of plasticity - dead units, weight magnitude, effective rank - without replacing anything, so it cannot change a run's trajectory. That is the measurement gym-CDA docs/25 recommends taking before switching the mechanism on, and it has now been taken: at ~20,000 optimiser steps on the scaled-down default config, at one seed, no loss of plasticity is detectable (gym-CDA docs/25 section 3.3, gym-CDA docs/16 section 16.16). Dead and saturated units sat at exactly zero throughout, and the network's rank on a FIXED corpus moved by +2.5% to -1.9% while cbp_batch_effective_rank fell 24.6% - the fall being the policy narrowing its own inputs, not the network losing capacity. That null does not carry to the 1e6-1e8 steps this project needs, which is the run still to do; read train/probe/rank.py across checkpoints rather than the batch metric when doing it.

### Scope

cbp_scope selects which units are replaceable. 'feedforward' means the hidden units of a two-Linear feed-forward block, plus the trunk's final hidden layer whose outgoing weights live in the pi/vf head rather than the encoder. Attention projections and embeddings are excluded: a unit there has no single well-defined outgoing weight matrix, and zeroing the outgoing weights is the whole reason a replacement does not disturb the function. 'recurrent' adds the hidden units of the top layer of an LSTM core (the lstm encoder has nothing replaceable under 'feedforward', so cbp_enabled there trained exactly like CBP off): a unit's incoming weights are its four gates' rows of weight_ih and weight_hh, and its outgoing ones are its column of weight_hh and of the head Linears (gym-CDA docs/29 S3-32, gym-CDA docs/25 section 3.4). Layers no optimiser trains are also skipped, which is what keeps it off the jepa encoder's EMA target trunk.

### Utility

cbp_utility selects the measure ranked against. 'contribution' is Nature equation 1, |h| * sum|w_out|. 'mean_corrected' subtracts the running mean activation, because gradient descent transfers the mean part of a removed unit's contribution to the consumer's bias anyway. 'overall' additionally divides by sum|w_in|, the adaptation utility: under Adam a unit with small incoming weights can change its function faster, so it is worth more than its contribution alone suggests. arXiv Appendix C ablates all three and finds 'overall' best, including on the reinforcement-learning task, which is why it is the default here.

### When it fires

cbp_fire_on selects when generate-and-test runs. 'adam_step' is after every optimiser step, which is what arXiv Algorithm 3 (Continual PPO) specifies. 'iteration' runs it once per training iteration instead, which cannot move the network underneath a log-prob recorded before it and so leaves PPO's exp(logp_new - logp_old) ratio undisturbed between minibatches. The papers use the per-step placement over 100M steps, so it is the default, but the conservative one is kept available.

### Compensating the mean

cbp_compensate_mean adds a replaced unit's mean contribution (its bias-corrected running mean activation times its outgoing weights) to the next layer's bias before the outgoing weights are zeroed, so the output does not jump (gym-CDA docs/29 R-13). Without it a saturated tanh unit, which 'overall' utility ranks lowest because its activation barely varies, is replaced first and takes its constant contribution with it: on a constructed case one replacement moved the outputs by 8-10 standard deviations. Off by default: the papers zero the outgoing weights and leave the bias alone, and gym-CDA docs/16 section 16.13 measured the real run that way, so switching it on is a different experiment. Run train.compare with it on and off before changing the default. Feed-forward layers only; a recurrent layer has no single consumer bias.
