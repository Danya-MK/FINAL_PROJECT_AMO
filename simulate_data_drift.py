import argparse
import pandas as pd
import numpy as np

FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", 
            "pH", "sulphates", "alcohol"]

def apply_drift(
    path_in: str,
    path_out: str | None,
    shift_alcohol: float,
    scale_density: float,
    noise_ph: float,
    seed: int,
):
    """
    Apply data drift to wine quality dataset.
    Simulates gradual changes in wine properties over time.
    """
    df = pd.read_csv(path_in)
    for c in FEATURES + ["target"]:
        if c not in df.columns:
            raise ValueError(f"{path_in} must contain column '{c}'")

    rng = np.random.default_rng(seed)
    df2 = df.copy()

    # дрейф только по фичам (target НЕ трогаем)
    if shift_alcohol != 0:
        df2["alcohol"] = df2["alcohol"] + shift_alcohol

    if scale_density != 1.0:
        df2["density"] = df2["density"] * scale_density

    if noise_ph != 0:
        df2["pH"] = df2["pH"] + rng.normal(0, noise_ph, size=len(df2))

    out = path_out or path_in
    df2.to_csv(out, index=False)
    print("Drift applied:", path_in, "->", out)
    print("Params:", dict(
        shift_alcohol=shift_alcohol,
        scale_density=scale_density,
        noise_ph=noise_ph
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="path_in", default="data/current.csv")
    p.add_argument("--out", dest="path_out", default=None, help="if not set -> overwrite input")
    p.add_argument("--shift-alcohol", type=float, default=0.5)
    p.add_argument("--scale-density", type=float, default=1.02)
    p.add_argument("--noise-ph", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    apply_drift(
        path_in=args.path_in,
        path_out=args.path_out,
        shift_alcohol=args.shift_alcohol,
        scale_density=args.scale_density,
        noise_ph=args.noise_ph,
        seed=args.seed
    )