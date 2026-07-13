from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.sample_voxel_animation import (
    _downsample_volume,
    _face_vertices,
    _render_frame,
    _resolve_label,
)


class VoxelAnimationTests(unittest.TestCase):
    def test_label_can_be_selected_by_id_or_name(self) -> None:
        names = ["chair_0", "chair_1", "sofa_0"]
        self.assertEqual(_resolve_label("1", names), 1)
        self.assertEqual(_resolve_label("SOFA_0", names), 2)
        with self.assertRaisesRegex(ValueError, "Unknown subtype"):
            _resolve_label("bed_0", names)

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
                title="raw",
                render_resolution=8,
                image_size=320,
                elev=30,
                azim=-60,
            ),
            _render_frame(
                filtered,
                removed=removed,
                title="detected",
                render_resolution=8,
                image_size=320,
                elev=30,
                azim=-60,
            ),
            _render_frame(
                filtered,
                title="filtered",
                render_resolution=8,
                image_size=320,
                elev=30,
                azim=-60,
            ),
        ]
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


if __name__ == "__main__":
    unittest.main()
