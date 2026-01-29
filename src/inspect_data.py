# src/inspect_data.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

def inspect(path):
    print("=== Inspecting", path.name, "===")
    df = pd.read_csv(path, nrows=5)
    print("Columns:", list(df.columns))
    print("Dtypes:")
    print(df.dtypes)
    print("\nFirst rows:")
    print(df.head())
    print("\nShape:", pd.read_csv(path).shape)  # full shape
    print("="*40, "\n")

if __name__ == "__main__":
    inspect(train_path)
    inspect(test_path)
