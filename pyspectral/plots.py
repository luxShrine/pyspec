from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, uniform_filter, zoom
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction

from pyspectral.config import REF_PHE, ArrayF, ArrayF32, PlotType
from pyspectral.core import Cube
from pyspectral.modeling.predict import ClassicalPredict, FoldPlot, PredictData

# -- figure count helper


def _get_figure_size(number_plots: int) -> tuple[int, int]:
    """Returns number of rows, by number of columns."""
    MAX_COLUMNS_PER_LINE = 4
    if number_plots <= MAX_COLUMNS_PER_LINE:
        # fits all on one row
        return 1, number_plots

    near_max_col = max(1, MAX_COLUMNS_PER_LINE - 1)
    if (number_plots % near_max_col) == 0:
        # fits an even amount minus one of max
        rows = int(number_plots / near_max_col)
        return rows, near_max_col

    # if above checks fail, just return sensible default, round up
    rows = (number_plots / MAX_COLUMNS_PER_LINE).__ceil__()
    return rows, MAX_COLUMNS_PER_LINE


# -- Boundary

# Binary operations reference:
# x ^ y ; set each bit to 1 if only one of the bits are 1
# x | y ; set each bit to 1 if one of the bits are 1


@dataclass
class Boundary:
    mask: ArrayF
    mask_type: str
    boundary: ArrayF = field(init=False)

    def __post_init__(self) -> None:
        """Create thin boundary of a binary mask."""
        er = binary_erosion(self.mask, structure=np.ones((3, 3), bool))
        self.boundary = self.mask ^ er  # pyright: ignore[reportOperatorIssue]

    @classmethod
    def create_hysteresis_mask(
        cls, r_map: ArrayF, low_q: float = 0.5, high_q: float = 0.7
    ) -> Boundary:
        if low_q >= high_q:
            raise ValueError(f"{low_q=} should be less than {high_q=}")
        # sanitize & thresholds
        r_map = np.nan_to_num(r_map.astype("float32"), copy=False)  # replace NaN/inf
        low_thresh = np.quantile(r_map, low_q)
        high_thresh = np.quantile(r_map, high_q)
        if not (low_thresh < high_thresh):
            # ensure a strict gap if quantiles somehow overlap
            eps = 1e-6
            high_thresh = low_thresh + eps

        mask = r_map >= low_thresh
        seed = r_map >= high_thresh

        # morphological reconstruction by dilation
        rec = reconstruction(
            seed.astype(np.uint8), mask.astype(np.uint8), method="dilation"
        )
        return cls(mask=rec.astype(bool), mask_type="hysteresis")

    @classmethod
    def create_otsu_mask(cls, r_map: ArrayF) -> Boundary:
        """Otsu threshold → boolean mask."""
        threshold = threshold_otsu(r_map.astype(np.float32))
        return cls(mask=(r_map >= threshold), mask_type="otsu")


# -- Ratio


def window_avg(
    y: ArrayF32, wl: ArrayF, center: float, halfwidth: float = 10.0
) -> ArrayF:
    """Average intensity in a small window around 'center' (in same units as wl)."""
    m = (wl >= center - halfwidth) & (wl <= center + halfwidth)
    return y[..., m].mean(axis=-1, dtype=np.float32)  # type: ignore


def ratio_map(
    cube_proc: Cube,
    wl: ArrayF,
    num_center: float,
    den_center: float,
    halfwidth: float,
    smooth_px: int,
) -> ArrayF:
    """
    cube_proc: (H,W,M) processed or normalized spectra
    wl: (M,)
    Returns R: (H,W) scalar ratio map.
    """
    num = window_avg(cube_proc.get(), wl, num_center, halfwidth)  # (H,W)
    den = window_avg(cube_proc.get(), wl, den_center, halfwidth) + 1e-8
    ratio = num / den
    if smooth_px > 0:
        ratio = uniform_filter(ratio, size=smooth_px)
    return ratio


def safe_den_center(
    den_center: float,
    stats: PredictData,
    *,
    fallback: float = REF_PHE,
    atol: float = 1.0,
) -> float:
    ref = stats.num_center
    if abs(den_center - ref) <= atol:
        return fallback
    return den_center


# -- Comparison/overlay


def rmse_per_pixel(cube_a: Cube | ArrayF32, cube_b: Cube | ArrayF32) -> ArrayF32:
    if isinstance(cube_a, Cube):
        cube_a = cube_a.get()
    if isinstance(cube_b, Cube):
        cube_b = cube_b.get()
    return np.sqrt(((cube_a - cube_b) ** 2).mean(axis=-1, dtype=np.float32))  # type: ignore


def overlay_boundary(base_img: ArrayF, boundary: ArrayF, title: str, ax: Axes) -> None:
    """
    Overlay boundary over base image.
    base_img: (H,W) float image to show (ratio or mean intensity)
    boundary: (H,W) bool edge map
    """
    ax.imshow(base_img, cmap="gray")
    overlay = np.zeros((*boundary.shape, 4), float)
    overlay[boundary] = (1.0, 0.0, 0.0, 0.9)  # red edges
    ax.imshow(overlay)
    ax.set_title(title)
    ax.set_axis_off()


def compare_boundaries(plot_data: PredictData, up: bool = False) -> dict[str, float]:
    """
    Build a ratio boundary.

    up: bool default: False, Whether or not to zoom into the array using spline interpolation.
    """
    den = safe_den_center(den_center=REF_PHE, stats=plot_data)
    # Make ratio maps + masks + boundaries
    ratio_map_cube = partial(
        ratio_map,
        wl=plot_data.wl,
        num_center=plot_data.num_center,
        den_center=den,
        halfwidth=26.0,
        smooth_px=1,
    )
    r_raw = ratio_map_cube(plot_data.raw_cube)
    r_proc = ratio_map_cube(plot_data.proc_cube)
    r_pred = ratio_map_cube(plot_data.pred_cube)

    # masks try hysteresis and maybe fallback to Otsu
    boundary_raw = Boundary.create_hysteresis_mask(r_raw)
    boundary_proc = Boundary.create_hysteresis_mask(r_proc)
    boundary_pred = Boundary.create_hysteresis_mask(r_pred)

    prc_pred_boundary_xor = boundary_proc.boundary ^ boundary_pred.boundary  # pyright: ignore[reportOperatorIssue]
    raw_pred_boundary_xor = boundary_raw.boundary ^ boundary_pred.boundary  # pyright: ignore[reportOperatorIssue]
    rp_boundary_xor = boundary_raw.boundary ^ boundary_proc.boundary  # pyright: ignore[reportOperatorIssue]

    # per-pixel RMSE, across bands
    prc_pred_rmse_map = rmse_per_pixel(plot_data.proc_cube, plot_data.pred_cube)
    raw_pred_rmse_map = rmse_per_pixel(plot_data.raw_cube, plot_data.pred_cube)
    rp_rmse_map = rmse_per_pixel(plot_data.raw_cube, plot_data.proc_cube)

    metrics: dict[str, float] = {}

    # figure
    scale = 20 if up else 1
    upscale = partial(zoom, zoom=scale, order=0)
    imgs = [
        (
            upscale(r_raw),
            upscale(boundary_proc.boundary),
            "Raw vs ML: ratio + boundary",
        ),
        (upscale(r_pred), upscale(boundary_pred.boundary), "ML OOF: ratio + boundary"),
        (
            upscale(r_raw),
            upscale(boundary_proc.boundary),
            "Raw vs Processed: ratio + boundary",
        ),
        (
            upscale(r_pred),
            upscale(prc_pred_boundary_xor),
            "Processed-Pred Disagreement (XOR)",
        ),
        (
            upscale(r_pred),
            upscale(raw_pred_boundary_xor),
            "Raw-Pred Disagreement (XOR)",
        ),
        (upscale(r_raw), upscale(rp_boundary_xor), "Raw-Processed Disagreement (XOR)"),
        (upscale(prc_pred_rmse_map), None, "Proccessed-Pred Per-pixel RMSE"),
        (upscale(rp_rmse_map), None, "Raw-Processed Per-pixel RMSE"),
        (upscale(raw_pred_rmse_map), None, "Raw-Pred Per-pixel RMSE"),
    ]

    n = len(imgs)
    r, c = _get_figure_size(n)
    figsize = (4.2 * n, 4)
    _fig, axes = plt.subplots(r, c, dpi=200)
    axes: list[Axes] = np.atleast_1d(axes).flatten().tolist()
    for ax, (base, edge, title) in zip(axes, imgs):
        ax.imshow(base, cmap="gray")
        if edge is not None:
            overlay = np.zeros((*edge.shape, 4), float)
            overlay[edge] = (1, 0, 0, 0.9)
            ax.imshow(overlay)
        ax.set_title(title, {"fontsize": 8}, loc="center", wrap=True)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return metrics


type FilteredPlotDataTest = list[tuple[None, ArrayF, str]]
type FilteredPlotDataTrain = list[tuple[ArrayF, None, str]]
type FilteredPlotDataAll = list[tuple[ArrayF, ArrayF, str]]


def filter_plot_type(
    plot_type: PlotType, fp_data: list[FoldPlot]
) -> FilteredPlotDataTest | FilteredPlotDataTrain | FilteredPlotDataAll:
    threshold = 10  # remove plots greater than some threshold
    match plot_type:
        case PlotType.TEST:
            plot_data = ((None, d.test_loss, d.label) for d in fp_data)
            plot_data = list(filter(lambda p: np.all(p[1] <= threshold), plot_data))
        case PlotType.TRAIN:
            plot_data = ((d.train_loss, None, d.label) for d in fp_data)  # type: ignore
            plot_data = list(filter(lambda p: np.all(p[0] <= threshold), plot_data))
        case PlotType.BOTH:
            plot_data = ((d.train_loss, d.test_loss, d.label) for d in fp_data)  # type: ignore
            plot_data = list(filter(lambda p: np.all(p[0] <= threshold), plot_data))
    return plot_data


def _plot_classic_line(
    ax: Axes, elasticnet_rmse: float, pcr_rmse: float, epochs: int
) -> None:
    """Add horizontal line for PCR/ElasticNet."""
    ax.hlines(
        elasticnet_rmse,
        colors="r",
        label=f"Enet (RMSE={elasticnet_rmse:.3f})",
        xmin=1,
        xmax=epochs,
    )
    ax.hlines(
        pcr_rmse,
        colors="b",
        label=f"PCR (RMSE={pcr_rmse:.3f})",
        xmin=1,
        xmax=epochs,
    )


def plot_loss_comparison_different_axes(
    fp_data: list[FoldPlot],
    class_error: ClassicalPredict,
    plot_type: PlotType = PlotType.TEST,
) -> None:
    fig, ax1 = plt.subplots(dpi=150)
    ax2 = ax1.twinx()
    a1_color = "tab:olive"
    a2_color = "tab:purple"
    ax1.set_ylabel("LSM RMSE", color=a1_color)
    ax1.tick_params(axis="y", labelcolor=a1_color)
    ax2.set_ylabel("LRSM RMSE", color=a2_color)
    ax2.tick_params(axis="y", labelcolor=a2_color)

    # plt.margins(y=0.01) # Adds % padding based on data range
    # plt.yscale("log")
    # ax1.set_ylim(0, 1)

    plot_data = filter_plot_type(plot_type, fp_data)

    epochs = len(fp_data[-1].test_loss)

    for i, (tr_l, te_l, lbl) in enumerate(plot_data):
        if "lsm" in lbl:
            axes = ax1
            s_color = a1_color
        else:
            axes = ax2
            s_color = a2_color
        style = "-" if (i % 2) == 0 else "--"
        style = "-." if (i % 3) == 0 else style
        style = ":" if (i % 4) == 0 else style
        if tr_l is not None:
            axes.plot(tr_l, style, label=f"train: {lbl}", color=s_color)
        if te_l is not None:
            axes.plot(te_l, style, label=f"test: {lbl}", color=s_color)

    _plot_classic_line(ax1, class_error.elasticnet_rmse, class_error.pcr_rmse, epochs)
    ax1.legend(loc="center left", bbox_to_anchor=(-1.5, 0.6), fontsize="small")
    ax2.legend(loc="center left", bbox_to_anchor=(1.5, 0.6), fontsize="small")
    fig.show()


def plot_loss_comparison(
    fp_data: list[FoldPlot],
    class_error: ClassicalPredict,
    plot_type: PlotType = PlotType.TEST,
) -> None:
    _fig, ax1 = plt.subplots(dpi=150)
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("Epoch Count")
    ax1.tick_params(axis="y")

    # plt.margins(y=0.01)# Adds % padding based on data range
    # plt.yscale("log")
    # ax1.set_ylim(43, 45)

    plot_data = filter_plot_type(plot_type, fp_data)
    # TODO: average across training and test for LSM and LRSM

    epochs = len(fp_data[-1].test_loss)
    er = range(1, epochs + 1)

    for i, (tr_l, te_l, lbl) in enumerate(plot_data):
        style = "-" if (i % 2) == 0 else "--"
        style = "-." if (i % 3) == 0 else style
        style = ":" if (i % 4) == 0 else style
        if tr_l is not None:
            ax1.plot(er, tr_l, style, label=f"train: {lbl}")
        if te_l is not None:
            ax1.plot(er, te_l, style, label=f"test: {lbl}")

    _plot_classic_line(ax1, class_error.elasticnet_rmse, class_error.pcr_rmse, epochs)

    ax1.legend(loc="center left", bbox_to_anchor=(1.15, 0.6), fontsize="small")
    plt.show()


# def plot_confusion(
#     data: PredictData,
# ) -> None:
#     from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
#
#     cm = confusion_matrix(data.train_pred, data.test_pred, labels=data.labels )
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.labels)
#     disp.plot()
#     plt.show()
