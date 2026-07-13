from __future__ import annotations

import unittest

import torch

from ddiff.postprocessing.voxels import filter_voxel_components


class VoxelPostprocessingTests(unittest.TestCase):
    def test_largest_filter_removes_floating_components(self) -> None:
        sample = torch.zeros((1, 8, 8, 8), dtype=torch.long)
        sample[0, 1:3, 1:3, 1:3] = 1
        sample[0, 6, 6, 6] = 1

        filtered, stats = filter_voxel_components(sample, mode="largest", connectivity=6)

        self.assertEqual(int(filtered.sum()), 8)
        self.assertEqual(int(filtered[0, 6, 6, 6]), 0)
        self.assertEqual(stats[0].components, 2)
        self.assertEqual(stats[0].original_voxels, 9)
        self.assertEqual(stats[0].kept_voxels, 8)
        self.assertEqual(stats[0].removed_voxels, 1)
        self.assertEqual(filtered.dtype, sample.dtype)
        self.assertEqual(filtered.device, sample.device)

    def test_connectivity_controls_diagonal_contact(self) -> None:
        sample = torch.zeros((1, 3, 3, 3), dtype=torch.uint8)
        sample[0, 0, 0, 0] = 1
        sample[0, 1, 1, 1] = 1

        filtered_6, stats_6 = filter_voxel_components(sample, connectivity=6)
        filtered_26, stats_26 = filter_voxel_components(sample, connectivity=26)

        self.assertEqual(stats_6[0].components, 2)
        self.assertEqual(int(filtered_6.sum()), 1)
        self.assertEqual(stats_26[0].components, 1)
        self.assertEqual(int(filtered_26.sum()), 2)

    def test_none_mode_reports_components_without_modifying_sample(self) -> None:
        sample = torch.zeros((1, 4, 4, 4), dtype=torch.long)
        sample[0, 0, 0, 0] = 1
        sample[0, 3, 3, 3] = 1

        filtered, stats = filter_voxel_components(sample, mode="none")

        self.assertTrue(torch.equal(filtered, sample))
        self.assertEqual(stats[0].components, 2)
        self.assertEqual(stats[0].removed_voxels, 0)

    def test_empty_sample_remains_empty(self) -> None:
        sample = torch.zeros((2, 4, 4, 4), dtype=torch.long)

        filtered, stats = filter_voxel_components(sample)

        self.assertTrue(torch.equal(filtered, sample))
        self.assertEqual([stat.components for stat in stats], [0, 0])
        self.assertEqual([stat.removed_voxels for stat in stats], [0, 0])


if __name__ == "__main__":
    unittest.main()
