import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import json, random
    import matplotlib.pyplot as plt
    from pathlib import Path
    import importlib

    SEED = 42
    random.seed(SEED); np.random.seed(SEED)

    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        torch = None
    return importlib, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Inspection
    """)
    return


@app.cell
def _(np, plt):
    from pyspectral.data.io import SpectraPair, read_pairs
    from pyspectral.data.preprocessing import PeakNormConfig, GlobalPeakNorm, ALS, PreConfig
    from pyspectral.config import DATA_DIR

    DATA = DATA_DIR
    ANN  = DATA / "tissue-alginate maps511.csv"
    peak_cfg = PeakNormConfig(mode=GlobalPeakNorm())
    pre_none = PreConfig(baseline=None)
    pre_poly = PreConfig.make_poly(7, 3)
    pre_ALS = PreConfig.make_als()

    rows = read_pairs( ANN, DATA)
    pairs_none, arts_none = SpectraPair.from_annotations(rows, peak_cfg=peak_cfg, pre_config=pre_none)
    pairs_def, arts_def = SpectraPair.from_annotations(rows, pre_config=pre_poly)
    pairs_als, arts_als = SpectraPair.from_annotations(rows, pre_config=pre_ALS) 

    X_none = pairs_none.X_raw.astype(np.float32)   
    Y_none  = pairs_none.Y_proc.astype(np.float32)  
    X_def = pairs_def.X_raw.astype(np.float32)   # (N,C)
    Y_def = pairs_def.Y_proc.astype(np.float32)  # (N,C)
    X_als = pairs_als.X_raw.astype(np.float32)   
    Y_als  = pairs_als.Y_proc.astype(np.float32)  

    plt.plot(X_none.mean(0), label="Raw mean"); plt.plot(Y_none.mean(0), label="LabSpec mean");
    plt.plot(X_def.mean(0), label="Default Raw mean"); plt.plot(Y_def.mean(0), label="Default LabSpec mean");
    plt.plot(X_als.mean(0), label="ALS Raw mean"); plt.plot(Y_als.mean(0), label="ALS LabSpec mean");
    plt.legend(); plt.title("Mean spectra Normalization Comparison"); plt.show()
    return ANN, DATA, arts_als, arts_def, pairs_als, pairs_def, pairs_none


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baselines
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classical Baseline
    """)
    return


@app.cell
def _(pairs_als, pairs_def, pairs_none):
    from sklearn.model_selection import KFold
    from sklearn.metrics import root_mean_squared_error as RMSE
    import pyspectral.result.predict as predict

    cv = KFold(n_splits=4, shuffle=True, random_state=42)

    def classic(pairs, Y, lbl):
        x, y = pairs.X_raw, pairs.Y_proc
        yhat_diag  = predict.diagonal_affine_predict(x, y, cv)                   
        yhat_pcr   = predict.pcr_predict(x, y, cv)
        yhat_enet  = predict.multitask_elasticnet_predict(x, y, cv)
        print(f"{lbl} Diagonal affine (OOF) RMSE: {RMSE(Y, yhat_diag):.4f}")
        print(f"{lbl} PCR (OOF) RMSE:             {RMSE(Y, yhat_pcr):.4f}")
        print(f"{lbl} ElasticNet (OOF) RMSE:      {RMSE(Y, yhat_enet):.4f}")
        print("-"*12)
        return yhat_diag, yhat_pcr, yhat_enet

    yhat_d_def, yhat_p_def, yhat_e_def = classic(pairs_def, pairs_none.Y_proc, "default")
    yhat_d_a, yhat_p_a, yhat_e_a = classic(pairs_als, pairs_none.Y_proc,"ALS")
    return (
        RMSE,
        predict,
        yhat_d_a,
        yhat_d_def,
        yhat_e_a,
        yhat_e_def,
        yhat_p_a,
        yhat_p_def,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train a low-rank linear neural network mapper & cache OOF preds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summarize models
    """)
    return


@app.cell
def _(arts_als, arts_def, importlib, pairs_als, pairs_def, pairs_new_norm):
    import pyspectral.modeling.train as tr
    from functools import partial
    importlib.reload(tr)

    train_nn = partial(tr.cv_train_model, 
        n_splits=4,
        epochs=10, 
        lr=2e-4,
        rank=12,
        verbose=True)

    oof_stats_def = train_nn(
        spectral_pairs=pairs_def,
        arts=arts_def,
    )
    oof_stats_nnorm = train_nn(
        spectral_pairs=pairs_new_norm,
        arts=arts_def,
    )
    oof_stats_als = train_nn(
        spectral_pairs=pairs_als,
        arts=arts_als,
    )

    print(f"{oof_stats_def=}") 
    print(f"{oof_stats_nnorm=}") 
    print(f"{oof_stats_als=}") 
    LR_pred_def = oof_stats_def.pred_orig  # (N,C) in original units
    LR_pred_nnorm = oof_stats_nnorm.pred_orig  
    LR_pred_als = oof_stats_als.pred_orig
    return (
        LR_pred_als,
        LR_pred_def,
        LR_pred_nnorm,
        oof_stats_als,
        oof_stats_def,
        oof_stats_nnorm,
    )


@app.cell
def _(
    LR_pred_als,
    LR_pred_def,
    LR_pred_nnorm,
    RMSE,
    Y,
    yhat_d_a,
    yhat_d_def,
    yhat_d_n,
    yhat_e_a,
    yhat_e_def,
    yhat_e_n,
    yhat_p_a,
    yhat_p_def,
    yhat_p_n,
):
    import polars as pl
    rank=12
    df = pl.DataFrame(
        {
            "model": ["def diag_affine", "def PCR", "def ElasticNet", f"def LowRank_r{rank}", 
            "nnorm diag_affine", "nnorm PCR", "nnorm ElasticNet", f"nnorm LowRank_r{rank}",
            "ALS diag_affine", "ALS PCR", "ALS ElasticNet", f"ALS LowRank_r{rank}"
            ],
            "oof_rmse": [
                RMSE(Y, yhat_d_def), RMSE(Y, yhat_p_def), RMSE(Y, yhat_e_def), RMSE(Y, LR_pred_def),
                RMSE(Y, yhat_d_n), RMSE(Y, yhat_p_n), RMSE(Y, yhat_e_n), RMSE(Y, LR_pred_nnorm),
                RMSE(Y, yhat_d_a), RMSE(Y, yhat_p_a), RMSE(Y, yhat_e_a), RMSE(Y, LR_pred_als)
                ]
        }
    ).sort("oof_rmse") # sort table in terms of rmse
    df
    return


@app.cell
def _(H, LR_pred_als, LR_pred_def, LR_pred_nnorm, W, Y, np, plt):
    from pyspectral.plots import rmse_per_pixel
    x_label = r"Band ($\text{cm}^{-1}$)"
    per_band_rmse_def = np.sqrt(((Y - LR_pred_def)**2).mean(axis=0))
    per_band_rmse_nnorm = np.sqrt(((Y - LR_pred_nnorm)**2).mean(axis=0))
    per_band_rmse_als = np.sqrt(((Y - LR_pred_als)**2).mean(axis=0))
    plt.figure(figsize=(10,3));plt.plot(per_band_rmse_als, label="ALS baseline"); plt.plot(per_band_rmse_def , label="default"); plt.plot(per_band_rmse_nnorm, label="new norm")
    plt.title("Per-band RMSE"); plt.xlabel(x_label); plt.ylabel("RMSE"); plt.yscale("log"); plt.legend(); plt.show()

    rmse_px_def = rmse_per_pixel(Y, LR_pred_def).reshape(H,W)
    rmse_px_nnorm = rmse_per_pixel(Y, LR_pred_nnorm).reshape(H,W)
    rmse_px_als = rmse_per_pixel(Y, LR_pred_als).reshape(H,W)
    plt.imshow(rmse_px_def, cmap="magma"); plt.colorbar(); plt.title("Per-pixel RMSE — default"); plt.show()
    plt.imshow(rmse_px_nnorm); plt.colorbar(); plt.title("Per-pixel RMSE — new norm"); plt.show()
    plt.imshow(rmse_px_als); plt.colorbar(); plt.title("Per-pixel RMSE — ALS"); plt.show()
    return


@app.cell
def _(
    ANN,
    DATA,
    importlib,
    oof_stats_als,
    oof_stats_def,
    oof_stats_nnorm,
    predict,
):
    import pyspectral.plots as plot
    importlib.reload(predict)
    importlib.reload(plot)
    scene_idx = 0  # pick a scene 
    pdata_def = predict.predict_cube(scene_idx, oof_stats_def, base_dir=DATA, csv_path=ANN)
    metrics_def = plot.compare_boundaries(pdata_def)
    pdata_nnorm = predict.predict_cube(scene_idx, oof_stats_nnorm, base_dir=DATA, csv_path=ANN)
    metrics_nnorm = plot.compare_boundaries(pdata_nnorm)
    pdata_als = predict.predict_cube(scene_idx, oof_stats_als, base_dir=DATA, csv_path=ANN)
    metrics_als = plot.compare_boundaries(pdata_als)
    return


if __name__ == "__main__":
    app.run()
