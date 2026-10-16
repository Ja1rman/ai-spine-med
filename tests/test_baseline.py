import numpy as np

from ai_spine_med.baseline import isotropic_inplane


def test_resampling_preserves_physical_spacing_and_slice_axis():
    # A linear intensity ramp in physical row coordinates must remain linear
    # after resampling; simply rounding zoom dimensions changes this spacing.
    rows = np.arange(10, dtype=np.float32) * 0.9
    volume = np.broadcast_to(rows[:, None, None], (10, 12, 3)).copy()
    volume[..., 1] += 100
    result, spacing, record = isotropic_inplane(volume, (0.9, 0.6))
    assert spacing == 0.6
    assert result.shape[-1] == 3
    np.testing.assert_allclose(result[:10, 0, 0], np.arange(10) * 0.6, atol=1e-5)
    np.testing.assert_allclose(result[:10, 0, 1], np.arange(10) * 0.6 + 100, atol=1e-5)
    assert max(abs(value) for value in record["last_center_extent_delta_mm"]) <= 0.3 + 1e-9
