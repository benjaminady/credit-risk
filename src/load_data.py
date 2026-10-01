"""Loading, feature selection and the train/test split."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data" / "credit_default.csv"
TARGET = "default.payment.next.month"

# Excluded from the feature set. In the United States the Equal Credit
# Opportunity Act and Regulation B prohibit using sex, marital status or age in
# a credit decision. See DECISIONS.md for what the exclusion costs.
PROTECTED = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


def load(drop_protected: bool = True) -> pd.DataFrame:
    """Return the dataset with ID removed and, by default, the protected
    attributes removed as well."""
    df = pd.read_csv(DATA).drop(columns=["ID"])
    if drop_protected:
        df = df.drop(columns=PROTECTED)
    return df


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 7):
    """Stratified split, so the default rate is identical in train and test."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


if __name__ == "__main__":
    df = load()
    Xtr, Xte, ytr, yte = split(df)
    print(f"rows {len(df):,}   features {Xtr.shape[1]}")
    print(f"default rate: overall {df[TARGET].mean():.4f}  "
          f"train {ytr.mean():.4f}  test {yte.mean():.4f}")
