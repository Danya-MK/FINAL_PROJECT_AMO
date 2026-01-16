import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request

# Wine Quality features for prediction
FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", 
            "pH", "sulphates", "alcohol"]

def make_data_wine(
    out_train: str,
    out_current: str,
    test_size: float,
    random_state: int,
    quality_threshold: int = 6,
):
    """
    Load Wine Quality dataset and split into train/current sets.
    Predict: quality > threshold (binary classification)
    """
    # Download wine quality dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    wine_path = "data/winequality-red.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(wine_path), exist_ok=True)
    
    if not os.path.exists(wine_path):
        print(f"Downloading wine quality dataset from {url}...")
        urllib.request.urlretrieve(url, wine_path)
    
    df = pd.read_csv(wine_path, sep=";")
    
    # Normalize column names: strip whitespace and convert to lowercase
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Update FEATURES to lowercase to match column names
    features_lower = [f.lower() for f in FEATURES]
    
    # Create binary target: quality >= threshold
    df["target"] = (df["quality"] >= quality_threshold).astype(int)
    
    # Keep only required features and target
    df = df[features_lower + ["target"]].dropna().copy()
    
    # Rename columns back to original names for consistency
    rename_dict = {fl: fo for fl, fo in zip(features_lower, FEATURES)}
    df = df.rename(columns=rename_dict)
    
    train_df, current_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"]
    )
    
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    train_df.to_csv(out_train, index=False)
    current_df.to_csv(out_current, index=False)
    
    print("Saved:", out_train, out_current)
    print("Train size:", len(train_df), "Current size:", len(current_df))
    print("Train target mean:", train_df["target"].mean(), "Current target mean:", current_df["target"].mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-train", default="data/train.csv")
    p.add_argument("--out-current", default="data/current.csv")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--quality-threshold", type=int, default=6, help="quality >= threshold -> target=1")
    args = p.parse_args()

    make_data_wine(
        out_train=args.out_train,
        out_current=args.out_current,
        test_size=args.test_size,
        random_state=args.random_state,
        quality_threshold=args.quality_threshold
    )