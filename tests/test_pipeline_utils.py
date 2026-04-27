from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iris.pipeline import _build_theta_grid, _gabor_kernel_1d, _safe_image_id, _subject_id_from_input_path, _subject_id_from_name, best_hamming_distance, IrisTemplate, Stage5Params


class SubjectIdTests(unittest.TestCase):
    def test_subject_id_with_underscore_suffix(self) -> None:
        self.assertEqual(_subject_id_from_name("subject_01"), "subject")

    def test_subject_id_with_r_suffix(self) -> None:
        self.assertEqual(_subject_id_from_name("S2001R09"), "S2001")

    def test_subject_id_with_trailing_digits(self) -> None:
        self.assertEqual(_subject_id_from_name("fionar1"), "fionar")

    def test_subject_id_from_parent_folder(self) -> None:
        base = Path("dataset")
        image = base / "person_07" / "left_01.bmp"
        self.assertEqual(_subject_id_from_input_path(base, image), "person_07")

    def test_subject_id_from_mmu_folder_structure(self) -> None:
        base = Path("MMU-Iris-Database")
        image = base / "13" / "left" / "lec_1.bmp"
        self.assertEqual(_subject_id_from_input_path(base, image), "13_left")

    def test_safe_image_id_uses_relative_path(self) -> None:
        base = Path("dataset")
        image = base / "person_07" / "left_01.bmp"
        self.assertEqual(_safe_image_id(base, image), "person_07_left_01")


class GaborKernelTests(unittest.TestCase):
    def test_gabor_kernel_has_nonzero_imaginary_part(self) -> None:
        kernel = _gabor_kernel_1d(np.pi / 128.0, angular_samples=128)
        self.assertGreater(float(np.max(np.abs(np.imag(kernel)))), 0.0)

    def test_theta_grid_excludes_top_and_bottom_sectors(self) -> None:
        params = Stage5Params(
            radial_samples_per_band=16,
            upper_angle_exclusion_deg=35.0,
            lower_angle_exclusion_deg=20.0,
            radial_bands=8,
            angular_samples=128,
            save_debug_images=False,
        )
        theta_deg = (np.rad2deg(_build_theta_grid(params)) % 360.0)
        eps = 1e-3
        self.assertFalse(np.any((theta_deg > 70.0 + eps) & (theta_deg < 110.0 - eps)))
        self.assertFalse(np.any((theta_deg > 235.0 + eps) & (theta_deg < 305.0 - eps)))


class MatchingTests(unittest.TestCase):
    def test_best_hamming_distance_detects_difference(self) -> None:
        code_a = np.zeros((1, 4, 2), dtype=np.uint8)
        code_b = np.zeros((1, 4, 2), dtype=np.uint8)
        code_b[0, 0, 0] = 1
        mask = np.ones_like(code_a, dtype=np.uint8)
        a = IrisTemplate("a.jpg", "A", code_a, mask)
        b = IrisTemplate("b.jpg", "B", code_b, mask)
        distance, _, valid_bits = best_hamming_distance(a, b, max_shift=0)
        self.assertEqual(valid_bits, 8)
        self.assertGreater(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
