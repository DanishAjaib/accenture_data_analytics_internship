import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and visualizations
    """
    
    def __init__(self, model: Any, preprocessor: Any):
        """
        Initialize evaluator with trained model and preprocessor
        
        Args:
            model: Trained model pipeline
            preprocessor: Fitted preprocessing pipeline
        """
        self.model = model
        self.preprocessor = preprocessor
        
    def generate_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, output_dict=True
            ),
            'feature_importances': self._get_feature_importances()
        }
    
    def _get_feature_importances(self) -> pd.DataFrame:
        """Extract feature importances if available"""
        try:
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                importances = self.model.named_steps['classifier'].feature_importances_
                features = self._get_feature_names()
                return pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
        except Exception as e:
            logger.warning(f"Could not get feature importances: {str(e)}")
        return None
        
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            # Handle one-hot encoded features
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                return list(self.preprocessor.get_feature_names_out())
            return list(self.model.feature_names_in_)
        except:
            return [f'feature_{i}' for i in range(self.model.n_features_in_)]
    
    def plot_confusion_matrix(self, cm: np.ndarray, classes: List[str]) -> plt.Figure:
        """Generate confusion matrix visualization"""
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.close()
        return fig
    
    def plot_roc_curve(self, y_test: pd.Series, y_proba: np.ndarray) -> plt.Figure:
        """Generate ROC curve visualization"""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        plt.close()
        return fig
    
    def plot_feature_importances(self, importances: pd.DataFrame) -> plt.Figure:
        """Generate feature importance visualization"""
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', 
                   data=importances.head(10), ax=ax)
        ax.set_title('Top 10 Feature Importances')
        plt.close()
        return fig


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: Any
) -> Tuple[Dict, Dict]:
    """
    Run complete evaluation pipeline
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        preprocessor: Fitted preprocessing pipeline
        
    Returns:
        Tuple of (metrics dictionary, plots dictionary)
    """
    evaluator = ModelEvaluator(model, preprocessor)
    report = evaluator.generate_report(X_test, y_test)
    
    plots = {
        'confusion_matrix': evaluator.plot_confusion_matrix(
            report['confusion_matrix'],
            classes=['Not Complete', 'Complete']
        ),
        'roc_curve': evaluator.plot_roc_curve(
            y_test,
            model.predict_proba(X_test)[:, 1]
        )
    }
    
    if report['feature_importances'] is not None:
        plots['feature_importances'] = evaluator.plot_feature_importances(
            report['feature_importances']
        )
    
    return report, plots