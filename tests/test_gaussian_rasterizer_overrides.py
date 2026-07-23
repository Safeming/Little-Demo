import pytest
import torch


class _FakeGaussian:
    def __init__(self):
        self.get_xyz = torch.zeros((3, 3), dtype=torch.float32)
        self.get_opacity = torch.full((3, 1), 0.5, dtype=torch.float32)


def test_resolve_raster_opacity_uses_model_default():
    from gaussian_renderer import resolve_raster_opacity

    gaussian = _FakeGaussian()
    assert resolve_raster_opacity(gaussian, None) is gaussian.get_opacity


def test_resolve_raster_opacity_accepts_matching_override():
    from gaussian_renderer import resolve_raster_opacity

    gaussian = _FakeGaussian()
    override = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    assert resolve_raster_opacity(gaussian, override) is override


def test_resolve_raster_opacity_rejects_wrong_point_count():
    from gaussian_renderer import resolve_raster_opacity

    gaussian = _FakeGaussian()
    with pytest.raises(ValueError, match="point count"):
        resolve_raster_opacity(gaussian, torch.ones((2, 1), dtype=torch.float32))


def test_rasterize_gaussians_exposes_optional_opacity_override():
    import inspect
    from gaussian_renderer import rasterize_gaussians

    signature = inspect.signature(rasterize_gaussians)
    assert signature.parameters["opacities_precomp"].default is None
