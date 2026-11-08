"""Run one real training iteration and print what CBP's hooks saw.

For each generate-and-test pass (`_cbp_run`): the step, the module, the layer and
the number of rows its statistics average over; for each loss computation, the
module and `loss_mask.sum()` out of the batch rows. The two side by side show
forwards the hooks should not count (#11) and padded rows they should mask (#10).

    python .claude/skills/cbp-verify-finding/scripts/hook_rows.py [--encoder lstm]
        [--scope recurrent] [--minibatch 64] [--fire-on adam_step] [--layers 2]

Runs the repository's own `tests/conftest.py` TEST_CFG, `metrics_only` so the
weights do not move. About 15 s on CPU.
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))

import ray  # noqa: E402
from conftest import TEST_CFG, cbp  # noqa: E402

from gym_continuousDoubleAuction.train.train import TrainConfig, build_config  # noqa: E402

from cda_cbp import learner as cbp_learner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="mlp")
    parser.add_argument("--scope", default="feedforward")
    parser.add_argument("--minibatch", type=int, default=TEST_CFG["minibatch_size"])
    parser.add_argument("--fire-on", default="adam_step")
    parser.add_argument("--layers", type=int, default=2, help="layers printed per module")
    args = parser.parse_args()

    original_run = cbp_learner.CBPLearnerMixin._cbp_run

    def logged_run(self):
        for module_id, layers in self._cbp_layers.items():
            for layer in layers[: args.layers]:
                stats = self._cbp_stats.get((module_id, layer.name))
                rows = None if stats is None else int(stats["rows"])
                print(f"step {self._cbp_updates} {module_id} {layer.name} rows={rows}")
        return original_run(self)

    cbp_learner.CBPLearnerMixin._cbp_run = logged_run

    settings = dict(metrics_only=True, scope=args.scope, fire_on=args.fire_on)
    cfg = dict(TEST_CFG, minibatch_size=args.minibatch, encoder_type=args.encoder)
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
    try:
        ppo_config, _cb = build_config(TrainConfig(**cfg, **cbp(**settings)))
        algo = ppo_config.build_algo()
        learner = algo.learner_group._learner
        base_loss = type(learner).compute_loss_for_module

        def logged_loss(self, *, module_id, config, batch, fwd_out):
            mask = batch.get("loss_mask")
            if mask is not None:
                print(f"loss {module_id} loss_mask={int(mask.sum())}/{mask.numel()}")
            return base_loss(self, module_id=module_id, config=config, batch=batch, fwd_out=fwd_out)

        type(learner).compute_loss_for_module = logged_loss
        algo.train()
        algo.stop()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
