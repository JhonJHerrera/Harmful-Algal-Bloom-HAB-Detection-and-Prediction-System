import pandas as pd

def map_predictions_to_labels(y_pred, label_map=None):
    """
    Map numeric predictions to string class labels.

    Parameters:
    - y_pred: array-like of numeric predictions (e.g. [0, 1, 0])
    - label_map: dict mapping ints to strings (e.g. {0: 'Low', 1: 'High'})

    Returns:
    - pandas.Series with string labels
    """
    if label_map is None:
        label_map = {0: 'Low', 1: 'High'}
    return pd.Series(y_pred).map(label_map)

def generate_future_labels(df: pd.DataFrame, horizon: int = 1, chl_column="CHL_NN", threshold=10.0):
    """
    Generate binary classification labels (Low/High) for a future prediction horizon.

    Parameters:
    - df: DataFrame with at least a 'CHL_NN' column
    - horizon: number of steps ahead to shift (e.g., 1 for t+7, 2 for t+15, etc.)
    - chl_column: column name with chlorophyll values
    - threshold: numeric threshold to classify as Low or High

    Returns:
    - pandas.Series of 'Low' or 'High' labels (NaNs for trailing rows)
    """
    future_values = df[chl_column].shift(-horizon)
    labels = future_values.apply(
        lambda x: 'Low' if pd.notnull(x) and x < threshold else ('High' if pd.notnull(x) else None)
    )
    return labels