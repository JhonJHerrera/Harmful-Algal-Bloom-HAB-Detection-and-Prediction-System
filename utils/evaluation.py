import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

def _extract_metrics_array(report_dict):
    """
    Convert classification_report dict to a NumPy array.
    Rows: [Low], [High], [Accuracy]
    Columns: [precision, recall, f1-score, support]
    """
    metrics = []
    for label in ['Low', 'High']:
        row = report_dict.get(label, {})
        metrics.append([
            row.get('precision', np.nan),
            row.get('recall', np.nan),
            row.get('f1-score', np.nan),
            row.get('support', np.nan)
        ])
    accuracy = report_dict.get('accuracy', np.nan)
    metrics.append([accuracy, np.nan, np.nan, np.nan])
    return np.array(metrics)

def evaluate_classification(y_true, y_pred, horizon="7d", output_dir="test_result"):
    """
    Generate classification report and plots, and export as PDF + .npy array.
    """
    class_map = {'Low': 0, 'High': 1}
    y_true_num = pd.Series(y_true).map(class_map)
    y_pred_num = pd.Series(y_pred).map(class_map)

    output_path = os.path.join(output_dir, f"{horizon}_report")
    os.makedirs(output_path, exist_ok=True)

    # === Metrics PDF ===
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    pdf_metrics_path = os.path.join(output_path, f"metrics_{horizon}.pdf")
    with PdfPages(pdf_metrics_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, len(report_df) * 0.4 + 1))
        ax.axis('off')
        table = ax.table(cellText=report_df.round(3).values,
                         colLabels=report_df.columns,
                         rowLabels=report_df.index,
                         cellLoc='center',
                         loc='center')
        table.scale(1, 1.5)
        ax.set_title(f"Classification Report - {horizon}", fontsize=14, weight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # === Save as .npy ===
    metrics_array = _extract_metrics_array(report)
    metrics_path = os.path.join(output_path, f"metrics_array_{horizon}.npy")
    np.save(metrics_path, metrics_array)

    # === Plots PDF ===
    pdf_plots_path = os.path.join(output_path, f"plots_{horizon}.pdf")
    with PdfPages(pdf_plots_path) as pdf:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['Low', 'High'])
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=['Low', 'High'],
                    yticklabels=['Low', 'High'], ax=ax1)
        ax1.set_title(f"Confusion Matrix ({horizon})")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # Step plot
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.step(range(len(y_true_num)), y_true_num, label='Actual', color='blue', linewidth=2)
        ax2.step(range(len(y_pred_num)), y_pred_num, label='Predicted', color='orange', linestyle='--', linewidth=2)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Low', 'High'])
        ax2.set_ylabel(f'Class ({horizon})', fontsize=14)
        ax2.set_xlabel('Sample', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)

    print(f"Metrics PDF saved:  {pdf_metrics_path}")
    print(f"Plots PDF saved:    {pdf_plots_path}")
    print(f"Metrics array saved:{metrics_path}")
