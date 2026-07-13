from __future__ import annotations

import unittest

import torch

from ddiff.postprocessing.voxels import filter_voxel_components
from ddiff.train import (
    VoxelSampleBatch,
    _build_ema_model,
    _build_voxel_quality_reference,
    _evaluate_voxel_generation_quality,
    _learning_rate_for_step,
    _update_ema_model,
)
from ddiff.utils.checkpoints import load_sampling_weights


class TrainingImprovementTests(unittest.TestCase):
    def test_ema_interpolates_online_parameters(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema_model = _build_ema_model(model)
        with torch.no_grad():
            model.weight.fill_(2.0)

        _update_ema_model(ema_model, model, decay=0.5)

        self.assertTrue(torch.equal(ema_model.weight, torch.ones_like(ema_model.weight)))
        self.assertFalse(any(parameter.requires_grad for parameter in ema_model.parameters()))

        with torch.no_grad():
            model.weight.fill_(3.0)
        _update_ema_model(ema_model, model, decay=0.999, step=10, start_step=10)
        self.assertTrue(torch.equal(ema_model.weight, model.weight))

    def test_warmup_and_cosine_schedule_reach_configured_endpoints(self) -> None:
        cfg = {
            "train": {
                "lr": 1.0,
                "min_lr": 0.1,
                "lr_scheduler": "cosine",
                "warmup_steps": 2,
                "warmup_start_factor": 0.05,
            }
        }
        self.assertAlmostEqual(_learning_rate_for_step(cfg, 1, 10), 0.525)
        self.assertAlmostEqual(_learning_rate_for_step(cfg, 2, 10), 1.0)
        self.assertGreater(_learning_rate_for_step(cfg, 6, 10), 0.1)
        self.assertAlmostEqual(_learning_rate_for_step(cfg, 10, 10), 0.1)

    def test_sampling_prefers_ema_and_can_select_raw_weights(self) -> None:
        template = torch.nn.Linear(1, 1, bias=False)
        raw = torch.nn.Linear(1, 1, bias=False)
        ema = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            raw.weight.fill_(1.0)
            ema.weight.fill_(2.0)
        checkpoint = {"model": raw.state_dict(), "ema_model": ema.state_dict()}

        self.assertEqual(load_sampling_weights(template, checkpoint, "auto"), "ema_model")
        self.assertEqual(float(template.weight.item()), 2.0)
        self.assertEqual(load_sampling_weights(template, checkpoint, "model"), "model")
        self.assertEqual(float(template.weight.item()), 1.0)
        with self.assertRaisesRegex(ValueError, "does not contain EMA"):
            load_sampling_weights(template, {"model": raw.state_dict()}, "ema")

    def test_generation_score_prefers_clean_reference_shape(self) -> None:
        class Dataset:
            def __init__(self) -> None:
                self.x = torch.zeros((2, 6, 6, 6), dtype=torch.uint8)
                self.x[0, 1:5, 1:5, 1:5] = 1
                self.x[1, 1:5, 1:5, 1:5] = 1
                self.y = torch.tensor([0, 0])

        dataset = Dataset()
        reference = _build_voxel_quality_reference(
            dataset,
            num_labels=1,
            max_items_per_label=2,
        )
        exact = dataset.x[:1].clone()
        exact_filtered, exact_stats = filter_voxel_components(exact)
        exact_batch = VoxelSampleBatch(
            raw=exact,
            filtered=exact_filtered,
            labels=torch.tensor([0]),
            component_stats=exact_stats,
        )

        rough = exact.clone()
        rough[0, 5, 3, 3] = 1
        rough_filtered, rough_stats = filter_voxel_components(rough)
        rough_batch = VoxelSampleBatch(
            raw=rough,
            filtered=rough_filtered,
            labels=torch.tensor([0]),
            component_stats=rough_stats,
        )

        exact_metrics = _evaluate_voxel_generation_quality(exact_batch, reference, {})
        rough_metrics = _evaluate_voxel_generation_quality(rough_batch, reference, {})
        self.assertAlmostEqual(exact_metrics["score"], 0.0)
        self.assertGreater(rough_metrics["score"], exact_metrics["score"])
        self.assertLess(rough_metrics["nearest_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
