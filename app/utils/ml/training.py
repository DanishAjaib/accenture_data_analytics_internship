import pandas as pd
import numpy as np
import joblib
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles model training with hyperparameter tuning and cross-validation
    Supports multiple algorithms with configurable parameter grids
    """
    
    MODEL_REGISTRY = {
        'random_forest': RandomForestClassifier(),
        'logistic_regression': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        },
        'decision_tree': {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
    }
    
    def __init__(self, preprocessor: Any, model_type: str = 'random_forest'):
        """
        Initialize trainer with preprocessing pipeline and model type
        
        Args:
            preprocessor: Fitted preprocessing pipeline
            model_type: Type of model to train (random_forest, logistic_regression, etc.)
        """
        self.preprocessor = preprocessor
        self.model_type = model_type
        self.model = None
        self.best_params_ = None
        
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Tuple[Any, Dict]:
        """
        Train model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            scoring: Evaluation metric
            
        Returns:
            Tuple of (best model, best parameters)
        """
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', self.MODEL_REGISTRY[self.model_type])
            ])
            
            # Grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid=self.PARAM_GRIDS[self.model_type],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            logger.info(f"Training {self.model_type} with hyperparameter tuning...")
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")
            
            return self.model, grid_search.best_params_
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def save_model(self, path: str) -> None:
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No trained model available to save")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")


def train_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    model_types: List[str] = ['random_forest', 'xgboost'],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Train and evaluate multiple models to find the best performer
    
    Args:
        X: Full feature set
        y: Target labels
        preprocessor: Configured preprocessing pipeline
        model_types: List of model types to evaluate
        test_size: Size of test set
        random_state: Random seed
        
    Returns:
        Tuple of (best model, best parameters)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    best_score = -np.inf
    best_model = None
    best_params = None
    
    for model_type in model_types:
        try:
            trainer = ModelTrainer(preprocessor, model_type)
            model, params = trainer.train_model(X_train, y_train)
            
            # Evaluate on test set
            score = model.score(X_test, y_test)
            logger.info(f"{model_type} test accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
                
        except Exception as e:
            logger.warning(f"Failed to train {model_type}: {str(e)}")
    
    logger.info(f"Best model: {type(best_model.named_steps['classifier']).__name__}")
    logger.info(f"Best test accuracy: {best_score:.4f}")
    
    return best_model, best_params