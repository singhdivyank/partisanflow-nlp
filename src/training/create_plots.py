"""Create plots for evaluation metrics"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

_PLOT_DIR = Path("plots")

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> Path:
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    path = _PLOT_DIR / f"cm_{model_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_roc(y_true, y_scores, model_name: str, num_classes: int) -> Path:
    n = min(2, num_classes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = [axes] if n == 1 else axes
    fig.suptitle(f"ROC Curves - {model_name}", fontsize=14)

    for i in range(n):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        axes[i].plot(fpr, tpr, label=f"Class {i}")
        axes[i].plot([0, 1], [0, 1], "k--")
        axes[i].set_xlabel("FPR")
        axes[i].set_ylabel("TPR")
        axes[i].set_title(f"Class {i}")
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    path = _PLOT_DIR / f"roc_{model_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_pr(y_true, y_scores, model_name: str, num_classes: int) -> Path:
    n = min(2, num_classes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = [axes] if n == 1 else axes
    fig.suptitle(f"PR Curves - {model_name}", fontsize=14)

    for i in range(n):
        prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_scores[:, i])
        axes[i].plot(rec, prec, label=f"Class {i}")
        axes[i].set_xlabel("Recall")
        axes[i].set_ylabel("Precision")
        axes[i].set_title(f"Class {i}")
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    path = _PLOT_DIR / f"pr_{model_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
