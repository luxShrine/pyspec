from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, uniform_filter
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from pyspectral.config import REF_PHE
from pyspectral.core import Cube, TruePredPair
from pyspectral.result.class_ml import SVCPred
from pyspectral.result.compare import ClassicalPredict, FoldPlot, MLSVMPlot
from pyspectral.result.predict import MaskedValues
from pyspectral.types import (
    Arr1DF,
    ArrayF,
    ArrayF32,
    Numeric,
    PlotType,
    UnitFloat,
)

# -- types

type FilteredPlotDataTest = list[tuple[None, ArrayF, str]]
type FilteredPlotDataTrain = list[tuple[ArrayF, None, str]]
type FilteredPlotDataAll = list[tuple[ArrayF, ArrayF, str]]

# -- helpers


def build_axes(ax: Axes | None = None, dpi: int = 150):
    """Helper to only create axes consistently."""
    if ax is None:
        _, ax = plt.subplots(dpi=dpi)
    return ax


def _get_figure_size(number_plots: int) -> tuple[int, int]:
    """Returns number of rows, by number of columns, for use in conjunnction
    with matplotlib's subplots."""
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


def build_loss_plot(
    ax: Axes, *, loss_type: str | None = None, log_scale: bool = False
) -> Axes:
    loss_type = f"({loss_type})" if loss_type is not None else ""
    if log_scale:
        plt.yscale("log")
    ax.minorticks_on()
    ax.set_ylabel(f"Loss {loss_type}")
    ax.set_xlabel("Epochs (count)")
    ax.legend()
    ax.set_title("Train & Test Loss over Epochs")
    return ax


def window_avg(
    y: ArrayF32, wl: ArrayF, center: float, halfwidth: float = 10.0
) -> ArrayF:
    """Average intensity in a small window around 'center' (in same units as wl)."""
    m = (wl >= center - halfwidth) & (wl <= center + halfwidth)
    return y[..., m].mean(axis=-1, dtype=np.float32)  # type: ignore


def safe_den_center(
    den_center: float,
    num_center: float,
    *,
    fallback: float = REF_PHE,
    atol: float = 1.0,
) -> float:
    ref = num_center
    if abs(den_center - ref) <= atol:
        return fallback
    return den_center


def _plot_horiz_line(
    ax: Axes, elasticnet_rmse: float, pcr_rmse: float, epochs: int
) -> Axes:
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
    return ax


def rmse_per_pixel(cube_a: Cube | ArrayF32, cube_b: Cube | ArrayF32) -> ArrayF32:
    if isinstance(cube_a, Cube):
        cube_a = cube_a.get()
    if isinstance(cube_b, Cube):
        cube_b = cube_b.get()
    return np.sqrt(((cube_a - cube_b) ** 2).mean(axis=-1, dtype=np.float32))


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

    def overlay(self, base_img: ArrayF, title: str, ax: Axes) -> None:
        """
        Overlay boundary over base image.
        base_img: (H,W) float image to show (ratio or mean intensity)
        """
        ax.imshow(base_img, cmap="gray")
        overlay = np.zeros((*self.boundary.shape, 4), float)
        overlay[self.boundary] = (  # pyright: ignore[reportCallIssue, reportArgumentType]
            1.0,
            0.0,
            0.0,
            0.9,
        )  # red edges
        ax.imshow(overlay)
        ax.set_title(title)
        ax.set_axis_off()


# -- Ratio


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


# -- Comparison/overlay


def loss_comparison(
    fp_data: list[FoldPlot],
    class_error: ClassicalPredict,
    plot_type: PlotType = PlotType.TEST,
    *,
    ax: Axes | None = None,
) -> Axes:
    ax = build_axes(ax)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Epoch Count")
    ax.tick_params(axis="y")

    # plt.margins(y=0.01)# Adds % padding based on data range
    # plt.yscale("log")
    # ax1.set_ylim(43, 45)

    plot_data = filter_plot_type(plot_type, fp_data)

    epochs = len(fp_data[-1].test_loss)
    er = range(1, epochs + 1)

    for i, (tr_l, te_l, lbl) in enumerate(plot_data):
        style = "-" if (i % 2) == 0 else "--"
        style = "-." if (i % 3) == 0 else style
        style = ":" if (i % 4) == 0 else style
        if tr_l is not None:
            ax.plot(er, tr_l, style, label=f"train: {lbl}")
        if te_l is not None:
            ax.plot(er, te_l, style, label=f"test: {lbl}")

    _plot_horiz_line(ax, class_error.elasticnet_rmse, class_error.pcr_rmse, epochs)

    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.6), fontsize="small")
    return ax


def cm_fpr_fnr(cm: np.ndarray[tuple[int, int]]) -> tuple[float, float]:
    """Get the false positive and false negative rate from a confusion matrix array.

    cm: Must be a square two dimensional array.

    Returns:
        False positive rate, false negative rate.
    """
    # true false cm[0][0]
    # true true  cm[1][1]
    # false true cm[0][1]
    # false false cm[1][0]
    neg_total = cm[0].sum()
    pos_total = cm[1].sum()
    fpr = cm[0][1] / neg_total
    ffr = cm[1][0] / pos_total
    return fpr, ffr


def make_confusion_matrix(
    true_values: MaskedValues, pred_values: MaskedValues, threshold: UnitFloat
) -> np.ndarray[tuple[int, int]]:
    true_pn = true_values.get_positive_negative_mask()
    pred_pn = pred_values.get_positive_negative_mask(threshold=float(threshold))
    return confusion_matrix(true_pn, pred_pn)


def confusion(
    true_pred_pair: TruePredPair,
    *,
    labels: np.ndarray | None = None,
) -> ConfusionMatrixDisplay:
    """
    display_labels: ndarray of shape (n_classes,), labels for plot.
    If None, display labels are set from 0 to

    """
    cm = confusion_matrix(true_pred_pair.true, true_pred_pair.pred)
    return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)


# -- Pixel Plots -------------------------------------------------------


def plot_pixel_histogram(
    pos_pred: np.ndarray,
    neg_pred: np.ndarray,
    maybe_pred: np.ndarray,
    threshold: Numeric = 0.5,
    bins: int = 50,
    log_scale: bool = False,
    *,
    ax: Axes | None = None,
    analysis_type: str | None = None,
) -> Axes:
    ax = build_axes(ax)
    thresh_percent = UnitFloat(threshold)
    ax.hist(pos_pred, bins=bins, alpha=0.7, label="True pos pixels")
    ax.hist(maybe_pred, bins=bins, alpha=0.7, label="True class 1 (maybe)")
    ax.hist(neg_pred, bins=bins, alpha=0.7, label="True neg pixels")
    ax.axvline(thresh_percent, linestyle="--", label=f"threshold={thresh_percent}")
    ax.set_xlabel("Predicted positive probability")
    ax.set_ylabel("Pixel count")
    t = "Predicted P(pos) by true class (0/0.5/1)"
    if analysis_type is not None:
        t += f" for {analysis_type}"
    ax.set_title(t)
    ax.legend()

    if log_scale:
        ax.semilogy()

    return ax


# -- PCA Components -------------------------------------------------------


def pca_components(
    *,
    X: np.ndarray[tuple[int, ...]],
    y: np.ndarray[tuple[int]],
) -> Axes:
    fig, ax = plt.subplots(num=1, figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d", elev=-160, azim=165)

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],  # pyright: ignore[reportArgumentType]
        c=y,
        s=8,
    )

    ax.set(
        title="First three PCA dimensions",
        xlabel="1st Eigenvector",
        ylabel="2nd Eigenvector",
        zlabel="3rd Eigenvector",
    )
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    handles, label_vals = scatter.legend_elements()
    # the labels spit out latex symbols, so we must clean it out first
    label_vals = [
        float(lbl.replace("$\\mathdefault{", "").replace("}$", ""))
        for lbl in label_vals
    ]

    names = []
    for v in label_vals:
        if np.isclose(v, 0.0):
            names.append("negative")
        elif np.isclose(v, 0.5):
            names.append("maybe")
        elif np.isclose(v, 1.0):
            names.append("positive")
        else:
            names.append(f"class {v:g}")

    legend1 = ax.legend(handles, names, loc="upper right", title="Classes")
    ax.add_artist(legend1)

    return ax


def svc_2d_plane(
    svc: SVCPred,
    pca_feats: np.ndarray[tuple[int, ...]],
    y: np.ndarray,
    *,
    kernel_type: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "linear",
):
    # trained k-D model
    # mean in PC space, shape (k,)
    mu: Arr1DF = pca_feats.mean(axis=0)
    k: int = svc.k
    w: Arr1DF = svc.w
    b: float = svc.b

    # precompute contribution from PCs 3..k
    c0: np.ndarray = np.dot(w[2:], mu[2:]) + b

    # grid over PC1/PC2
    X0, X1 = pca_feats[:, 0], pca_feats[:, 1]
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    # decision function with PCs 3..k fixed at means
    zz = w[0] * xx + w[1] * yy + c0

    cmap = "coolwarm"
    _, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, zz > 0, cmap=cmap, alpha=0.2)  # regions
    ax.contour(xx, yy, zz, levels=[0], cmap=cmap)  # boundary f=0
    ax.scatter(X0, X1, c=y, s=20, cmap=cmap, edgecolor="k", alpha=0.6)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(f"Slice of {kernel_type} {k}-D SVM hyperplane in PC1–PC2 plane")
    return ax


def roc_auc(
    pred_pos: np.ndarray,
    pred_neg: np.ndarray,
    *,
    ax: Axes | None = None,
    count: int = 100,
) -> Axes:
    ax = build_axes(ax)
    threshold_arr = np.linspace(0, 1, count)

    false_positive_rate = []
    true_positive_rate = []

    total_count_p = pred_pos.size
    total_count_f = pred_neg.size
    for r in threshold_arr:
        pred_tp = np.where(pred_pos > r, 1, 0)
        pred_tp_count = np.count_nonzero(pred_tp)
        ratio_tp = (pred_tp_count) / total_count_p
        true_positive_rate.append(ratio_tp)

        pred_fp = np.where(pred_neg > r, 1, 0)
        pred_fp_count = np.count_nonzero(pred_fp)
        ratio_fp = (pred_fp_count) / total_count_f
        false_positive_rate.append(ratio_fp)

    ax.plot(false_positive_rate, true_positive_rate, label="ML learning curve)")
    ax.plot([0, 1], "r--", label="m=0.5 (No learning curve)")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    return ax


def do_MLSVM_plots(data: MLSVMPlot, threshold: float = 0.5) -> tuple[Axes, Axes]:
    threshold = UnitFloat(threshold)
    # ML first
    ml = data.ml_cmp
    plot_pixel_histogram(
        ml.pred.pos, ml.pred.neg, ml.pred.maybe, threshold=threshold, analysis_type="ML"
    )

    # PCA plots
    pca = data.pca_cmp
    plot_pixel_histogram(
        pca.pred.pos,
        pca.pred.neg,
        pca.pred.maybe,
        threshold=threshold,
        bins=25,
        analysis_type="PCA+SVM",
    )

    # keep positive & negative
    svc = data.svc_preds[0]
    X_pca = svc.X
    y_full = svc.y
    y_binary = (np.isclose(y_full, 1.0) | np.isclose(y_full, 2.0)).astype(int)

    axes_pca = pca_components(X=X_pca, y=y_full)
    axes_svc = svc_2d_plane(svc, X_pca, y_binary)

    # show IoU
    print(f"ML: IoU={ml.iou}")
    print(f"PCA: IoU={pca.iou}")

    return (axes_pca, axes_svc)
