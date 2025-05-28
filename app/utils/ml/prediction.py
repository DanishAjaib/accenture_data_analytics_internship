import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookingPredictor:
    """
    Handles making predictions with the trained model pipeline
    Includes pre-processing of new data and prediction interpretation
    """
    
    def __init__(self, model_pipeline: Pipeline, preprocessor: Any):
        """
        Initialize predictor with trained model and preprocessor
        
        Args:
            model_pipeline: Trained model pipeline
            preprocessor: Fitted preprocessing pipeline
        """
        self.model_pipeline = model_pipeline
        self.preprocessor = preprocessor
        
    def prepare_input(
        self,
        input_data: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare raw input data for prediction
        
        Args:
            input_data: Dictionary of raw input features
            
        Returns:
            Tuple of (processed DataFrame, list of feature names)
        """
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Apply same preprocessing
            processed_df = self.preprocessor.transform(input_df)
            
            # Get feature names
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                features = list(self.preprocessor.get_feature_names_out())
            else:
                features = [f'feature_{i}' for i in range(processed_df.shape[1])]
            
            return processed_df, features
            
        except Exception as e:
            logger.error(f"Error preparing input data: {str(e)}")
            raise
            
    def predict(
        self,
        input_data: Dict[str, Any],
        return_proba: bool = True
    ) -> Tuple[int, float]:
        """
        Make prediction on new data
        
        Args:
            input_data: Raw input features as dictionary
            return_proba: Whether to return probability
            
        Returns:
            Tuple of (prediction class, probability)
        """
        try:
            processed_df, _ = self.prepare_input(input_data)
            prediction = self.model_pipeline.predict(processed_df)[0]
            
            if return_proba:
                proba = self.model_pipeline.predict_proba(processed_df)[0][1]
                return prediction, proba
            return prediction, None
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
            
    def get_prediction_explanation(
        self,
        prediction: int,
        probability: float
    ) -> Dict[str, Any]:
        """
        Generate human-readable prediction explanation
        
        Args:
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            
        Returns:
            Dictionary with explanation components
        """
        return {
            'prediction': 'Booking Complete' if prediction == 1 else 'Booking Incomplete',
            'confidence': f"{probability:.1%}",
            'interpretation': (
                "The customer is likely to complete the booking" 
                if prediction == 1 else 
                "The customer is unlikely to complete the booking"
            )
        }


def load_predictor(model_path: str, preprocessor_path: str) -> BookingPredictor:
    """
    Load trained model and preprocessor from disk
    
    Args:
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        
    Returns:
        Initialized BookingPredictor instance
    """
    try:
        import joblib
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return BookingPredictor(model, preprocessor)
    except Exception as e:
        logger.error(f"Failed to load predictor: {str(e)}")
        raise