import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

class EvaluationReport:
    def __init__(self, y_true, y_pred, y_proba=None, feature_importances=None, feature_names=None):
        """
        y_true: array-like of true labels
        y_pred: array-like of predicted labels
        y_proba: array-like of predicted probabilities for positive class
        feature_importances: array-like of feature importance values
        feature_names: list of feature names corresponding to importances
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.feature_importances = feature_importances
        self.feature_names = feature_names

    def plot_confusion_matrix(self, normalize=False):
        cm = confusion_matrix(self.y_true, self.y_pred)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticklabels(['0', '1'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center')
        ax.set_title('Confusion Matrix')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_roc_curve(self):
        if self.y_proba is None:
            raise ValueError("y_proba is required for ROC curve")
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.grid(True)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_precision_recall_curve(self):
        if self.y_proba is None:
            raise ValueError("y_proba is required for precision-recall curve")
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_feature_importance(self, top_n=20):
        if self.feature_importances is None or self.feature_names is None:
            raise ValueError("feature_importances and feature_names are required for feature importance plot")
        importances = np.array(self.feature_importances)
        names = list(self.feature_names)
        indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[indices]
        top_names = [names[i] for i in indices]
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Top Feature Importances')
        fig.subplots_adjust(left=0.35)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def save_report(self, output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with PdfPages(output_path) as pdf:
            pdf.savefig(self.plot_confusion_matrix())
            if self.y_proba is not None:
                pdf.savefig(self.plot_roc_curve())
                pdf.savefig(self.plot_precision_recall_curve())
            if self.feature_importances is not None and self.feature_names is not None:
                pdf.savefig(self.plot_feature_importance())
        print(f"Evaluation report saved to: {output_path}")
