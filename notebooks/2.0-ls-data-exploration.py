import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell
def _():
    from pyspectral.config import DATA_DIR
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
    return DATA_DIR, np, plt


@app.cell
def _(DATA_DIR, np):
    text_path = DATA_DIR / "raw" / "tissue-alginate maps274BRawdata.txt"
    text_path_proc = DATA_DIR / "processed" / "tissue-alginate maps274BProcessed.txt"
    def read_data(text_path):
        with text_path.open("r") as f:
            wl = np.fromstring(next(line for line in f if line.strip()), sep="\t")
            rows = [np.fromstring(r_line, sep="\t") for r_line in f if r_line.strip()]
        xy_and_spectra = np.vstack(rows)
        return xy_and_spectra 
    raw_data = read_data(text_path)
    proc_data = read_data(text_path_proc)
    raw_data[:, 2:].shape, proc_data[:, 2:].shape
    return proc_data, raw_data


@app.cell
def _(plt, proc_data, raw_data):
    spectra_raw = raw_data[:, 2:]
    spectra_proc = proc_data[:, 2:]

    plt.plot(spectra_raw[0], label="raw") 
    plt.plot(spectra_proc[1], label="processed") 
    plt.legend(); plt.title("Raw spectra"); plt.show()
    return


@app.cell
def _(np, plt, proc_data, raw_data):
    def normalize(data):
        min = np.min(data)
        num = data - min
        den = np.max(data) - min
        return num / den
    spectra_raw_m = normalize(raw_data[:, 2:])
    spectra_proc_m = normalize(proc_data[:, 2:])

    plt.plot(spectra_raw_m[0], label="raw") 
    plt.plot(spectra_proc_m[1], label="processed") 
    plt.legend(); plt.title("Normalized spectra"); plt.show()
    return


if __name__ == "__main__":
    app.run()
