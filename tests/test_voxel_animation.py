from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.sample_voxel_animation import (
    _render_frame,
    _resolve_label,
    _surface_voxels,
)


class VoxelAnimationTests(unittest.TestCase):
    def test_label_can_be_selected_by_id_or_name(self) -> None:
        names = ["chair_0", "chair_1", "sofa_0"]
        self.assertEqual(_resolve_label("1", names), 1)
        self.assertEqual(_resolve_label("SOFA_0", names), 2)
        with self.assertRaisesRegex(ValueError, "Unknown subtype"):
            _resolve_label("bed_0", names)

    def test_surface_extraction_removes_solid_interior(self) -> None:
        occupied = np.ones((3, 3, 3), dtype=bool)
        surface = _surface_voxels(occupied)
        self.assertEqual(int(surface.sum()), 26)
        self.assertFalse(surface[1, 1, 1])

    def test_rendered_frames_can_be_written_as_gif(self) -> None:
        raw = torch.zeros((8, 8, 8), dtype=torch.long)
        raw[1:5, 1:5, 1:5] = 1
        raw[7, 7, 7] = 1
        filtered = raw.clone()
        filtered[7, 7, 7] = 0
        removed = raw.bool() & ~filtered.bool()

        frames = [
            _render_frame(raw, title="raw", max_points=1000, elev=24, azim=38),
            _render_frame(
                filtered,
                removed=removed,
                title="detected",
                max_points=1000,
                elev=24,
                azim=38,
            ),
            _render_frame(filtered, title="filtered", max_points=1000, elev=24, azim=38),
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
