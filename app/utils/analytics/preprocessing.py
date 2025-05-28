from typing import Tuple, List, Dict, Union, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler, 
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalerType(Enum):
    MINMAX = auto()
    STANDARD = auto()
    ROBUST = auto()

class EncoderType(Enum):
    ONEHOT = auto()
    ORDINAL = auto()
    LABEL = auto()

class DataPreprocessor:
    """
    Enhanced data preprocessing pipeline with configurable:
    - Scaler selection (MinMax, Standard, Robust)
    - Encoder selection (OneHot, Ordinal, Label)
    - Outlier handling strategies
    - Missing value imputation
    - Feature transformation options
    """
    
    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        binary_cols: Optional[List[str]] = None,
        scaler_type: ScalerType = ScalerType.MINMAX,
        encoder_type: EncoderType = EncoderType.ONEHOT,
        handle_outliers: bool = True,
        outlier_method: str = 'cap',
        log_transform: bool = True,
        impute_missing: bool = True
    ):
        """
        Initialize preprocessor with flexible configuration
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            binary_cols: List of binary column names (optional)
            scaler_type: Type of scaler to use (MINMAX, STANDARD, ROBUST)
            encoder_type: Type of encoder to use (ONEHOT, ORDINAL, LABEL)
            handle_outliers: Whether to handle outliers
            outlier_method: 'cap' or 'remove'
            log_transform: Whether to apply log transformation
            impute_missing: Whether to impute missing values
        """

        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols or []
        self.scaler_type = scaler_type
        self.encoder_type = encoder_type
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.log_transform = log_transform
        self.impute_missing = impute_missing
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize preprocessing components based on configuration"""
        # Initialize scaler
        if self.scaler_type == ScalerType.MINMAX:
            self.scaler = MinMaxScaler()
        elif self.scaler_type == ScalerType.STANDARD:
            self.scaler = StandardScaler()
        elif self.scaler_type == ScalerType.ROBUST:
            self.scaler = RobustScaler()
        
        # Initialize encoder
        if self.encoder_type == EncoderType.ONEHOT:
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif self.encoder_type == EncoderType.ORDINAL:
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        elif self.encoder_type == EncoderType.LABEL:
            self.encoder = LabelEncoder()  # Note: For single target column
            
        # Initialize imputer
        self.imputer = SimpleImputer(strategy='median')
        
        self.fitted = False
    
    def _validate_input(self, df: pd.DataFrame):
        """Validate input DataFrame"""
        missing_cols = [col for col in self.numerical_cols + self.categorical_cols 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numerical columns"""
        if not self.impute_missing:
            return df
            
        df_processed = df.copy()
        df_processed[self.numerical_cols] = self.imputer.fit_transform(
            df_processed[self.numerical_cols]
        )
        logger.info("Imputed missing values in numerical columns")
        return df_processed
    
    def _process_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers based on configured method"""
        if not self.handle_outliers:
            return df
            
        df_processed = df.copy()
        
        for col in self.numerical_cols:
            if col not in df.columns:
                continue
                
            q1 = df[col].quantile(0.05)
            q3 = df[col].quantile(0.95)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if self.outlier_method == 'cap':
                df_processed[col] = np.clip(
                    df_processed[col], 
                    lower_bound, 
                    upper_bound
                )
                logger.info(f"Capped outliers in column: {col}")
            elif self.outlier_method == 'remove':
                mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                df_processed = df_processed[mask].copy()
                logger.info(f"Removed outliers from column: {col}")
                
        return df_processed
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation if enabled"""
        if not self.log_transform:
            return df
            
        df_transformed = df.copy()
        for col in self.numerical_cols:
            if col in df.columns:
                df_transformed[col] = np.log1p(df_transformed[col])
                logger.info(f"Applied log transform to column: {col}")
        return df_transformed
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using configured scaler"""
        df_scaled = df.copy()
        
        if not self.fitted:
            self.scaler.fit(df_scaled[self.numerical_cols])
            self.fitted = True
            logger.info(f"Fitted {self.scaler_type.name} scaler")
            
        df_scaled[self.numerical_cols] = self.scaler.transform(
            df_scaled[self.numerical_cols]
        )
        logger.info("Applied feature scaling")
        return df_scaled
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features based on configured encoder"""
        df_encoded = df.copy()
        
        if self.encoder_type == EncoderType.LABEL:
            # LabelEncoder is only for target variables
            logger.warning("LabelEncoder is typically for target variables. Using Ordinal instead.")
            self.encoder = OrdinalEncoder()
            self.encoder_type = EncoderType.ORDINAL
            
        if self.categorical_cols:
            if not self.fitted:
                self.encoder.fit(df_encoded[self.categorical_cols])
                self.fitted = True
                logger.info(f"Fitted {self.encoder_type.name} encoder")
            
            encoded = self.encoder.transform(df_encoded[self.categorical_cols])
            
            if self.encoder_type == EncoderType.ONEHOT:
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=self.encoder.get_feature_names_out(self.categorical_cols)
                )
                df_encoded = pd.concat([df_encoded.drop(self.categorical_cols, axis=1), encoded_df], axis=1)
            else:
                for i, col in enumerate(self.categorical_cols):
                    df_encoded[col] = encoded[:, i]
        
        return df_encoded
    
    def full_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute complete preprocessing pipeline:
        1. Missing value imputation
        2. Outlier handling
        3. Feature transformation
        4. Feature scaling
        5. Categorical encoding
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Tuple of (processed DataFrame, list of feature names)
        """
        self._validate_input(df)
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Handle missing values
        df_processed = self._handle_missing_values(df)
        
        # Step 2: Process outliers
        df_processed = self._process_outliers(df_processed)
        
        # Step 3: Apply transformations
        df_processed = self._transform_features(df_processed)
        
        # Step 4: Scale features
        df_processed = self._scale_features(df_processed)
        
        # Step 5: Encode categoricals
        df_processed = self._encode_categoricals(df_processed)
        
        # Get final feature list
        features = [col for col in df_processed.columns 
                   if col not in ['booking_complete'] + self.binary_cols]
        
        logger.info("Preprocessing pipeline completed")
        return df_processed, features

    def get_preprocessor_pipeline(self) -> ColumnTransformer:
        """
        Create a scikit-learn ColumnTransformer pipeline for integration
        with sklearn workflows
        
        Returns:
            Configured ColumnTransformer
        """
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self.scaler)
        ])
        
        categorical_pipeline = Pipeline([
            ('encoder', self.encoder)
        ])
        
        return ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_cols),
            ('cat', categorical_pipeline, self.categorical_cols)
        ], remainder='passthrough')


def get_preprocessor(
    df: pd.DataFrame,
    target_col: str = 'booking_complete',
    scaler_type: ScalerType = ScalerType.MINMAX,
    encoder_type: EncoderType = EncoderType.ONEHOT,
    **kwargs
) -> DataPreprocessor:
    """
    Factory function to create a configured preprocessor
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        scaler_type: Type of scaler to use
        encoder_type: Type of encoder to use
        **kwargs: Additional arguments for DataPreprocessor
        
    Returns:
        Configured DataPreprocessor instance
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != target_col and df[col].nunique() > 2]
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_col]
    
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != target_col]
    
    return DataPreprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        scaler_type=scaler_type,
        encoder_type=encoder_type,
        **kwargs
    )