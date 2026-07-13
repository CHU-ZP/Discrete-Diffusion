from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ddiff.diffusion.categorical import CategoricalDiffusion
from scripts.sample_voxel_animation import (
    AnimationRenderTask,
    _downsample_volume,
    _face_vertices,
    _output_paths,
    _render_frame,
    _render_tasks,
    _resolve_label,
    _resolve_labels,
)


class VoxelAnimationTests(unittest.TestCase):
    def test_label_can_be_selected_by_id_or_name(self) -> None:
        names = ["chair_0", "chair_1", "sofa_0"]
        self.assertEqual(_resolve_label("1", names), 1)
        self.assertEqual(_resolve_label("SOFA_0", names), 2)
        with self.assertRaisesRegex(ValueError, "Unknown subtype"):
            _resolve_label("bed_0", names)

    def test_multiple_labels_can_be_selected_by_name_id_or_all(self) -> None:
        names = ["chair_0", "chair_1", "sofa_0"]
        self.assertEqual(_resolve_labels("chair_1, 2", names), [1, 2])
        self.assertEqual(_resolve_labels("all", names), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Duplicate subtype"):
            _resolve_labels("chair_0,0", names)

    def test_cube_faces_preserve_tensor_axis_order(self) -> None:
        coordinate = np.asarray([[1, 2, 3]])
        x_face = _face_vertices(coordinate, axis=0, sign=1)[0]
        z_face = _face_vertices(coordinate, axis=2, sign=1)[0]

        self.assertEqual(set(x_face[:, 0].tolist()), {2.0})
        self.assertEqual(set(x_face[:, 1].tolist()), {2.0, 3.0})
        self.assertEqual(set(x_face[:, 2].tolist()), {3.0, 4.0})
        self.assertEqual(set(z_face[:, 0].tolist()), {1.0, 2.0})
        self.assertEqual(set(z_face[:, 2].tolist()), {4.0})

    def test_noise_downsampling_preserves_shape_and_discrete_values(self) -> None:
        values = np.zeros((64, 64, 64), dtype=np.uint8)
        values[::2, ::2, ::2] = 1
        downsampled = _downsample_volume(values, 32)
        self.assertEqual(downsampled.shape, (32, 32, 32))
        self.assertTrue(set(np.unique(downsampled)).issubset({0, 1}))

    def test_rendered_frames_can_be_written_as_gif(self) -> None:
        raw = torch.zeros((8, 8, 8), dtype=torch.long)
        raw[1:5, 1:5, 1:5] = 1
        raw[7, 7, 7] = 1
        filtered = raw.clone()
        filtered[7, 7, 7] = 0
        removed = raw.bool() & ~filtered.bool()

        frames = [
            _render_frame(
                raw,
                render_resolution=8,
                image_size=320,
                elev=30,
                azim=-60,
            ),
            _render_frame(
                filtered,
                removed=removed,
                render_resolution=8,
                image_size=320,
                elev=30,
                azim=-60,
            ),
            _render_frame(
                filtered,
                render_resolution=8,
                image_size=320,
                elev=30,
                azim=-60,
            ),
        ]
        self.assertTrue(np.all(np.asarray(frames[0])[:20] == 255))
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "animation.gif"
            frames[0].save(
                output,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=[100, 200, 300],
                loop=0,
                disposal=2,
            )
            with Image.open(output) as image:
                self.assertEqual(image.n_frames, 3)

    def test_multiple_output_names_are_unique(self) -> None:
        cfg = {"output": {"sample_dir": "outputs/test"}}
        outputs = _output_paths("outputs/custom.gif", cfg, ["chair_0"], 3)
        self.assertEqual(
            [path.name for path in outputs],
            ["custom_sample_000.gif", "custom_sample_001.gif", "custom_sample_002.gif"],
        )

    def test_multiple_label_output_names_preserve_label_major_order(self) -> None:
        cfg = {"output": {"sample_dir": "outputs/test"}}
        outputs = _output_paths(
            "outputs/custom.gif",
            cfg,
            ["chair_0", "sofa_1"],
            2,
        )
        self.assertEqual(
            [path.name for path in outputs],
            [
                "custom_chair_0_sample_000.gif",
                "custom_chair_0_sample_001.gif",
                "custom_sofa_1_sample_000.gif",
                "custom_sofa_1_sample_001.gif",
            ],
        )

    def test_multiple_animations_render_in_parallel(self) -> None:
        raw = np.zeros((3, 8, 8, 8), dtype=np.uint8)
        raw[0, 1:3, 1:3, 1:3] = 1
        raw[1, 1:4, 1:4, 1:4] = 1
        raw[2, 1:5, 1:5, 1:5] = 1
        filtered = raw[-1].copy()
        removed = np.zeros_like(filtered)

        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = [
                AnimationRenderTask(
                    sample_index=index,
                    diffusion_frames=raw,
                    diffusion_steps=(2, 1, 0),
                    filtered_result=filtered,
                    removed_voxels=removed,
                    output=Path(temp_dir) / f"sample_{index}.gif",
                    frame_duration_ms=50,
                    hold_duration_ms=100,
                    render_resolution=8,
                    final_render_resolution=8,
                    image_size=240,
                    elev=30,
                    azim=-60,
                )
                for index in range(2)
            ]
            results = _render_tasks(tasks, worker_count=2)
            self.assertEqual([index for index, _ in results], [0, 1])
            self.assertTrue(all(path.exists() for _, path in results))

    def test_diffusion_chain_can_be_recorded_as_uint8(self) -> None:
        class ZeroModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x_t, t, y=None):
                return torch.zeros(
                    (x_t.shape[0], 2, *x_t.shape[1:]),
                    device=x_t.device,
                )

        diffusion = CategoricalDiffusion(2, 2, 0.1, 0.2)
        _, chain = diffusion.sample(
            ZeroModel(),
            (3, 3, 3),
            batch_size=2,
            return_chain=True,
            chain_steps=(2, 1, 0),
            chain_dtype=torch.uint8,
        )
        self.assertEqual(set(chain), {0, 1, 2})
        self.assertTrue(all(frame.dtype == torch.uint8 for frame in chain.values()))


if __name__ == "__main__":
    unittest.main()
