from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch

from ddiff.data.modelnet_voxel import (
    resolve_voxel_label_names,
    validate_voxel_conditioning_metadata,
)
from ddiff.postprocessing.voxels import filter_voxel_components
from ddiff.sample import (
    _build_conditioning_labels,
    _resolve_sampling_label_names,
    _save_voxel_samples,
    _validate_checkpoint_sampling_config,
)
from ddiff.train import _save_checkpoint, _validate_dataset_conditioning
from ddiff.visualization.voxels import save_voxel_grid


def subtype_metadata(num_labels: int = 12) -> dict[str, np.ndarray]:
    return {
        "subtype_names": np.asarray([f"shape_{index}" for index in range(num_labels)]),
        "subtype_class_ids": np.arange(num_labels, dtype=np.int64) // 3,
        "subtype_local_ids": np.arange(num_labels, dtype=np.int64) % 3,
        "subtype_counts": np.ones(num_labels, dtype=np.int64),
        "subtype_test_counts": np.ones(num_labels, dtype=np.int64),
        "subtype_centers": np.zeros((num_labels, 4), dtype=np.float32),
        "class_names": np.asarray(["chair", "sofa", "bed", "monitor"]),
    }


def voxel_config(num_labels: int = 12) -> dict:
    return {
        "dataset": {
            "name": "modelnet10_voxel",
            "conditional": True,
            "num_labels": num_labels,
            "cache_path": "not-used-in-this-test.npz",
        }
    }


class SamplingLabelTests(unittest.TestCase):
    def test_all_labels_covers_every_subtype_in_order(self) -> None:
        labels = _build_conditioning_labels(
            voxel_config(),
            12,
            "all",
            None,
            torch.device("cpu"),
            voxel_metadata=subtype_metadata(),
        )
        self.assertEqual(labels.tolist(), list(range(12)))

    def test_explicit_labels_repeat_without_changing_ids(self) -> None:
        labels = _build_conditioning_labels(
            voxel_config(),
            4,
            "3,7",
            None,
            torch.device("cpu"),
            voxel_metadata=subtype_metadata(),
        )
        self.assertEqual(labels.tolist(), [3, 7, 3, 7])

    def test_prior_sampling_uses_subtype_ids_from_metadata(self) -> None:
        metadata = subtype_metadata()
        metadata["subtype_counts"] = np.zeros(12, dtype=np.int64)
        metadata["subtype_counts"][7] = 10
        labels = _build_conditioning_labels(
            voxel_config(),
            5,
            None,
            None,
            torch.device("cpu"),
            voxel_metadata=metadata,
        )
        self.assertEqual(labels.tolist(), [7] * 5)

    def test_class_sampling_stays_inside_the_requested_class(self) -> None:
        labels = _build_conditioning_labels(
            voxel_config(),
            20,
            None,
            "monitor",
            torch.device("cpu"),
            voxel_metadata=subtype_metadata(),
        )
        self.assertTrue(set(labels.tolist()).issubset({9, 10, 11}))

    def test_all_labels_rejects_too_few_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot cover all 12 requested labels"):
            _build_conditioning_labels(
                voxel_config(),
                4,
                "all",
                None,
                torch.device("cpu"),
                voxel_metadata=subtype_metadata(),
            )

    def test_metadata_length_mismatch_is_rejected(self) -> None:
        metadata = subtype_metadata()
        metadata["subtype_counts"] = np.ones(11, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "subtype_counts describes 11 labels"):
            validate_voxel_conditioning_metadata(metadata, 12)

    def test_checkpoint_and_cache_mapping_mismatch_is_rejected(self) -> None:
        metadata = subtype_metadata()
        checkpoint = {"conditioning_label_names": [f"other_{index}" for index in range(12)]}
        with self.assertRaisesRegex(ValueError, "different conditioning-label mappings"):
            _resolve_sampling_label_names(checkpoint, metadata, 12)

    def test_sampling_config_mismatch_is_rejected_before_generation(self) -> None:
        cfg = voxel_config()
        cfg["dataset"].update(num_classes=2, shape=[64, 64, 64])
        cfg["diffusion"] = {
            "timesteps": 100,
            "schedule": "linear",
            "beta_start": 0.005,
            "beta_end": 0.15,
            "transition": "uniform",
        }
        cfg["model"] = {"name": "unet3d"}
        checkpoint_cfg = {
            **cfg,
            "diffusion": {**cfg["diffusion"], "beta_end": 0.2},
        }
        with self.assertRaisesRegex(ValueError, "diffusion.beta_end"):
            _validate_checkpoint_sampling_config(cfg, {"config": checkpoint_cfg})

    def test_voxel_grid_uses_names_and_renders_every_requested_item(self) -> None:
        metadata = subtype_metadata()
        names = resolve_voxel_label_names(metadata, 12)
        samples = torch.zeros((12, 2, 2, 2), dtype=torch.long)
        labels = torch.arange(12)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "grid.png"
            with patch("ddiff.visualization.voxels.plt.close"):
                save_voxel_grid(
                    samples,
                    output,
                    max_items=12,
                    labels=labels,
                    label_names=names,
                )
                figure = plt.gcf()
                self.assertEqual(len(figure.axes), 12)
                self.assertEqual([axis.get_title() for axis in figure.axes], names)
            plt.close(figure)
            self.assertTrue(output.exists())

    def test_raw_voxel_output_preserves_id_and_name_for_each_sample(self) -> None:
        names = resolve_voxel_label_names(subtype_metadata(), 12)
        raw_samples = torch.zeros((12, 3, 3, 3), dtype=torch.long)
        raw_samples[:, 0:2, 0:2, 0:2] = 1
        raw_samples[:, 2, 2, 2] = 1
        samples, component_stats = filter_voxel_components(raw_samples)
        labels = torch.arange(12)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "samples.npz"
            _save_voxel_samples(
                samples,
                labels,
                names,
                output,
                raw_samples=raw_samples,
                component_stats=component_stats,
                component_filter_mode="largest",
                component_connectivity=6,
            )
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data["labels"].tolist(), list(range(12)))
                self.assertEqual(data["sample_label_names"].tolist(), names)
                self.assertEqual(data["samples"].shape, (12, 3, 3, 3))
                self.assertEqual(data["raw_samples"].shape, (12, 3, 3, 3))
                self.assertEqual(data["removed_voxel_counts"].tolist(), [1] * 12)
                self.assertEqual(data["component_filter_mode"].item(), "largest")
                self.assertEqual(data["component_connectivity"].item(), 6)

    def test_checkpoint_preserves_the_training_label_mapping(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        names = resolve_voxel_label_names(subtype_metadata(), 12)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "checkpoint.pt"
            _save_checkpoint(
                model,
                optimizer,
                voxel_config(),
                10,
                output,
                conditioning_label_names=names,
            )
            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["conditioning_label_names"], names)

    def test_training_rejects_missing_conditioning_labels(self) -> None:
        class Dataset:
            y = torch.tensor([0, 1, 3])

            def __len__(self) -> int:
                return len(self.y)

        cfg = voxel_config(num_labels=4)
        with self.assertRaisesRegex(ValueError, r"no samples for conditioning labels \[2\]"):
            _validate_dataset_conditioning(cfg, Dataset())


if __name__ == "__main__":
    unittest.main()
