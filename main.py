import pandas as pd
from src.models.xgboost_model import load_model_and_scaler, predict_with_model
from utils.labeling import map_predictions_to_labels, generate_future_labels
from utils.evaluation import evaluate_classification

def run_prediction(horizon_label="7d", horizon_steps=1,
                   input_path="data/processed/test_input.csv", 
                   output_dir="test_result"):
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(horizon_label)

    # Load input features
    df = pd.read_csv(input_path)

    # Predict (exclude label if present)
    y_pred_raw = predict_with_model(model, scaler, df, drop_cols=['CHL_NN','date','latitude', 'longitude'])
    # Map numeric predictions to class labels
    y_pred_labels = map_predictions_to_labels(y_pred_raw)

    # Generate true labels for the corresponding horizon
    y_true_labels = generate_future_labels(df, horizon=horizon_steps).dropna()

    # Align predictions and true labels by index
    min_len = min(len(y_true_labels), len(y_pred_labels))
    y_true_aligned = y_true_labels.iloc[:min_len]
    y_pred_aligned = y_pred_labels.iloc[:min_len]

    # Evaluate
    evaluate_classification(
        y_true=y_true_aligned,
        y_pred=y_pred_aligned,
        horizon=horizon_label,
        output_dir=output_dir
    )


if __name__ == "__main__":
    run_prediction(horizon_label="7d", horizon_steps=1)