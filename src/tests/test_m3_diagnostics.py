"""Regression checks for M3 diagnostic rendering."""

import matplotlib
import pytest
import torch

from mvo.models.m3_selfsup_posedepth.diagnostics import _imshow_rgb

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.bool, torch.float16, torch.float32])
def test_rgb_diagnostic_renders_supported_dtype(dtype):
    image = torch.ones((3, 8, 8), dtype=dtype)
    figure, axes = plt.subplots()
    try:
        _imshow_rgb(axes, image)
        assert axes.images[0].get_array().dtype == "float32"
        figure.canvas.draw()
    finally:
        plt.close(figure)
