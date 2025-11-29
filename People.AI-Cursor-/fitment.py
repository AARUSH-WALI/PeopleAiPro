import pandas as pd
import numpy as np
import pickle
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class EnhancedFitmentScoringSystem:
    """
    Production-ready Enhanced Fitment Scoring System
    Bracket-relative scoring with proper pkl serialization for website integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.scaler = StandardScaler()
        self.institute_codes = {}
        self.historical_data = None
        self.bracket_stats = {}
        self.processed_candidates = {}
        
        self.config = config or {
            'use_tiered_scoring': True,
            'use_bracket_relative': True,
            'include_institute_bonus': True,
            'include_state_bonus': True,
            'personality_weight_fresher': 0.70,
            'technical_weight_fresher': 0.30,
            'personality_weight_experienced': 0.30,
            'technical_weight_experienced': 0.70,
        }
        
        self.brackets = {
            1: {'min': 0, 'max': 2, 'label': 'Entry Level (0-2 years)'},
            2: {'min': 2, 'max': 5, 'label': 'Junior (2-5 years)'},
            3: {'min': 5, 'max': 10, 'label': 'Mid-Level (5-10 years)'},
            4: {'min': 10, 'max': 100, 'label': 'Senior (10+ years)'}
        }
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate required columns exist"""
        required_cols = ['Longevity_Years', 'Average_Experience']
        return all(col in df.columns for col in required_cols)
    
    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and preprocess dataset"""
        if not self._validate_dataframe(df):
            raise ValueError("Missing required columns: Longevity_Years, Average_Experience")
        
        self.historical_data = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        if 'UG_Institute_Code' in df.columns:
            self.institute_codes = {
                'UG': set(df['UG_Institute_Code'].unique()),
                'PG': set(df['PG_Institute_Code'].unique()) if 'PG_Institute_Code' in df.columns else set(),
                'PHD': set(df['PHD_Institute_Code'].unique()) if 'PHD_Institute_Code' in df.columns else set()
            }
        
        return df
    
    def assign_experience_bracket(self, longevity_years: float) -> int:
        """Assign experience bracket based on longevity"""
        if longevity_years < 2:
            return 1
        elif longevity_years < 5:
            return 2
        elif longevity_years < 10:
            return 3
        else:
            return 4
    
    def classify_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify candidates into categories and brackets"""
        df = df.copy()
        df['Experience_Bracket'] = df['Longevity_Years'].apply(self.assign_experience_bracket)
        
        def get_category(row):
            if row['Longevity_Years'] < 2:
                return 'Fresher'
            elif row['Longevity_Years'] < 5:
                return 'Inexperienced'
            else:
                return 'Experienced'
        
        df['Category'] = df.apply(get_category, axis=1)
        self._calculate_bracket_statistics(df)
        
        print(f"\n{'='*80}")
        print("CANDIDATE CLASSIFICATION")
        print(f"{'='*80}")
        for bracket, info in self.brackets.items():
            count = len(df[df['Experience_Bracket'] == bracket])
            print(f"Bracket {bracket} ({info['label']}): {count} candidates")
        
        return df
    
    def _calculate_bracket_statistics(self, df: pd.DataFrame):
        """Calculate statistics for each bracket"""
        self.bracket_stats = {}
        technical_features = [
            'Longevity_Years', 'Average_Experience', 'TotalPatents', 'TotalPapers',
            'Workshops', 'Trainings', 'Achievements', 'Books'
        ]
        
        for bracket in sorted(df['Experience_Bracket'].unique()):
            bracket_data = df[df['Experience_Bracket'] == bracket]
            self.bracket_stats[bracket] = {}
            
            for feature in technical_features:
                if feature in bracket_data.columns and len(bracket_data) > 0:
                    self.bracket_stats[bracket][feature] = {
                        'max': float(bracket_data[feature].max()),
                        'min': float(bracket_data[feature].min()),
                        'mean': float(bracket_data[feature].mean()),
                        'median': float(bracket_data[feature].median()),
                        'std': float(bracket_data[feature].std() or 0)
                    }
    
    def calculate_tiered_longevity_score(self, longevity_years: float) -> float:
        """Tiered scoring for longevity"""
        if longevity_years >= 9:
            return 100.0
        elif longevity_years >= 7:
            return 90.0
        elif longevity_years >= 5:
            return 85.0
        elif longevity_years >= 3:
            return 70.0
        else:
            return 50.0
    
    def calculate_tiered_avgexp_score(self, avg_experience: float) -> float:
        """Tiered scoring for average experience"""
        if avg_experience > 7:
            return 100.0
        elif avg_experience > 5:
            return 90.0
        elif avg_experience > 3:
            return 75.0
        elif avg_experience >= 2:
            return 65.0
        else:
            return 50.0
    
    def calculate_technical_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical score with bracket-relative scoring"""
        df = df.copy()
        
        weights = {
            'Longevity_Years': 0.25,
            'Average_Experience': 0.25,
            'TotalPatents': 0.13,
            'TotalPapers': 0.13,
            'Workshops': 0.085,
            'Trainings': 0.065,
            'Achievements': 0.04,
            'Books': 0.02,
        }
        
        bonus_weights = {
            'State_J&K': 0.02,
            'UG_Institute': 0.02,
            'PG_Institute': 0.03,
            'PHD_Institute': 0.05
        }
        
        df['Technical_Score'] = 0.0
        df['Technical_Score_Breakdown'] = ''
        
        print(f"\n{'='*80}")
        print("CALCULATING TECHNICAL SCORES")
        print(f"{'='*80}")
        
        for bracket in sorted(df['Experience_Bracket'].unique()):
            bracket_mask = df['Experience_Bracket'] == bracket
            bracket_name = self.brackets[bracket]['label']
            print(f"\nProcessing Bracket {bracket} ({bracket_name})...")
            
            for idx, row in df[bracket_mask].iterrows():
                score = 0.0
                breakdown = []
                
                if self.config['use_tiered_scoring']:
                    longevity_score = self.calculate_tiered_longevity_score(row['Longevity_Years'])
                    avgexp_score = self.calculate_tiered_avgexp_score(row['Average_Experience'])
                    
                    score += longevity_score * weights['Longevity_Years']
                    score += avgexp_score * weights['Average_Experience']
                    
                    breakdown.append(f"Longevity: {longevity_score * weights['Longevity_Years']:.2f}")
                    breakdown.append(f"AvgExp: {avgexp_score * weights['Average_Experience']:.2f}")
                else:
                    longevity_norm = min(row['Longevity_Years'] / 15 * 100, 100)
                    avgexp_norm = min(row['Average_Experience'] / 10 * 100, 100)
                    
                    score += longevity_norm * weights['Longevity_Years']
                    score += avgexp_norm * weights['Average_Experience']
                
                for feature in ['TotalPatents', 'TotalPapers', 'Workshops', 'Trainings', 
                               'Achievements', 'Books']:
                    if feature in df.columns and feature in self.bracket_stats.get(bracket, {}):
                        bracket_mean = self.bracket_stats[bracket][feature]['mean']
                        bracket_max = self.bracket_stats[bracket][feature]['max']
                        
                        if self.config['use_bracket_relative'] and bracket_mean > 0:
                            normalized_val = min((row[feature] / bracket_mean) * 100, 100)
                        elif bracket_max > 0:
                            normalized_val = min((row[feature] / bracket_max) * 100, 100)
                        else:
                            normalized_val = 0.0
                        
                        contribution = normalized_val * weights.get(feature, 0)
                        score += contribution
                        breakdown.append(f"{feature}: {contribution:.2f}")
                
                if self.config['include_state_bonus'] and 'State_J&K' in df.columns:
                    if row['State_J&K'] == 1:
                        bonus = 100 * bonus_weights['State_J&K']
                        score += bonus
                        breakdown.append(f"State: {bonus:.2f}")
                
                if self.config['include_institute_bonus']:
                    for inst_type in ['UG_Institute', 'PG_Institute', 'PHD_Institute']:
                        if inst_type in df.columns and row[inst_type] == 1:
                            bonus = 100 * bonus_weights[inst_type]
                            score += bonus
                            breakdown.append(f"{inst_type}: {bonus:.2f}")
                
                df.loc[idx, 'Technical_Score'] = min(round(score, 2), 100.0)
                df.loc[idx, 'Technical_Score_Breakdown'] = '; '.join(breakdown)
            
            bracket_scores = df.loc[bracket_mask, 'Technical_Score']
            print(f"  Score Range: {bracket_scores.min():.1f} - {bracket_scores.max():.1f}")
            print(f"  Mean Score: {bracket_scores.mean():.1f}")
        
        return df
    
    def calculate_personality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate personality score with bracket-relative normalization"""
        df = df.copy()
        
        personality_weights = {
            'ExtroversionScore': 0.15,
            'AgreeablenessScore': 0.20,
            'ConscientiousnessScore': 0.30,
            'NeuroticismScore': -0.15,
            'OpennessToExperienceScore': 0.25
        }
        
        df['Personality_Score'] = 0.0
        
        print(f"\n{'='*80}")
        print("CALCULATING PERSONALITY SCORES")
        print(f"{'='*80}")
        
        for bracket in sorted(df['Experience_Bracket'].unique()):
            bracket_mask = df['Experience_Bracket'] == bracket
            bracket_name = self.brackets[bracket]['label']
            print(f"\nProcessing Bracket {bracket} ({bracket_name})...")
            
            raw_scores = []
            for idx, row in df[bracket_mask].iterrows():
                score = 0.0
                for trait, weight in personality_weights.items():
                    if trait in df.columns:
                        if trait == 'NeuroticismScore':
                            normalized = max(0, 40 - row[trait]) / 40 * 100
                        else:
                            normalized = min(row[trait] / 40 * 100, 100)
                        
                        score += normalized * abs(weight)
                
                raw_scores.append(score)
            
            if len(raw_scores) > 1:
                min_score = min(raw_scores)
                max_score = max(raw_scores)
                
                if max_score != min_score:
                    normalized_scores = [
                        ((s - min_score) / (max_score - min_score)) * 100 
                        for s in raw_scores
                    ]
                else:
                    normalized_scores = [50.0] * len(raw_scores)
            else:
                normalized_scores = [50.0]
            
            for i, idx in enumerate(df[bracket_mask].index):
                df.loc[idx, 'Personality_Score'] = round(normalized_scores[i], 2)
            
            bracket_scores = df.loc[bracket_mask, 'Personality_Score']
            print(f"  Score Range: {bracket_scores.min():.1f} - {bracket_scores.max():.1f}")
            print(f"  Mean Score: {bracket_scores.mean():.1f}")
        
        return df
    
    def calculate_fitment_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final fitment score with percentiles (Updated for >= 3 years technical weight)"""
        df = df.copy()
        
        print(f"\n{'='*80}")
        print("CALCULATING FITMENT SCORES")
        print(f"{'='*80}")
        
        # --- FIX START ---
        
        # 1. Define the condition for the Technical-Heavy weight (>= 3.0 years)
        is_technical_heavy = df['Longevity_Years'] >= 3.0
        
        # 2. Calculate scores using the two predefined weight schemes
        #    Score for Technical-Heavy (70% Technical / 30% Personality)
        score_technical_heavy = (self.config['technical_weight_experienced'] * df['Technical_Score'] + 
                                 self.config['personality_weight_experienced'] * df['Personality_Score'])
        
        #    Score for Personality-Heavy (70% Personality / 30% Technical)
        score_personality_heavy = (self.config['personality_weight_fresher'] * df['Personality_Score'] + 
                                   self.config['technical_weight_fresher'] * df['Technical_Score'])
        
        # 3. Apply the scores efficiently using numpy.where based on the 3.0 year threshold
        #    Note: This vectorized approach replaces the slow iterrows loop for score calculation.
        df['Fitment_Score'] = np.where(
            is_technical_heavy,
            score_technical_heavy,
            score_personality_heavy
        ).round(2)
        
        # --- FIX END ---
        
        df['Fitment_Percentile'] = 0.0
        df['Fitment_Category'] = ''
        
        for bracket in sorted(df['Experience_Bracket'].unique()):
            bracket_mask = df['Experience_Bracket'] == bracket
            bracket_scores = df.loc[bracket_mask, 'Fitment_Score']
            
            percentiles = bracket_scores.rank(pct=True) * 100
            df.loc[bracket_mask, 'Fitment_Percentile'] = percentiles.round(1)
            
            for idx in df[bracket_mask].index:
                percentile = df.loc[idx, 'Fitment_Percentile']
                if percentile >= 75:
                    df.loc[idx, 'Fitment_Category'] = 'Excellent'
                elif percentile >= 50:
                    df.loc[idx, 'Fitment_Category'] = 'Good'
                elif percentile >= 25:
                    df.loc[idx, 'Fitment_Category'] = 'Average'
                else:
                    df.loc[idx, 'Fitment_Category'] = 'Below Average'
            
            bracket_name = self.brackets[bracket]['label']
            print(f"\nBracket {bracket} ({bracket_name}):")
            print(f"  Score Range: {bracket_scores.min():.1f} - {bracket_scores.max():.1f}")
            print(f"  Mean Score: {bracket_scores.mean():.1f}")
            
            category_counts = df[bracket_mask]['Fitment_Category'].value_counts()
            for cat, count in category_counts.items():
                print(f"    {cat}: {count}")
        
        return df
        
    def process_fitment_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete end-to-end fitment scoring pipeline"""
        print(f"\n{'='*80}")
        print("ENHANCED FITMENT SCORING SYSTEM")
        print(f"{'='*80}")
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        df = self.load_data(df)
        df = self.classify_candidates(df)
        df = self.calculate_technical_score(df)
        df = self.calculate_personality_score(df)
        df = self.calculate_fitment_score(df)
        
        # Store for pkl export
        if 'Employee_Code' in df.columns or 'ID' in df.columns:
            id_col = 'Employee_Code' if 'Employee_Code' in df.columns else 'ID'
            for idx, row in df.iterrows():
                self.processed_candidates[str(row[id_col])] = row.to_dict()
        
        print(f"\n{'='*80}")
        print("SCORING COMPLETE!")
        print(f"{'='*80}")
        
        return df
    
    def save_to_pkl(self, filepath: str = 'fitment_scorer.pkl'):
        """Save scorer state to pickle file"""
        data = {
            'config': self.config,
            'brackets': self.brackets,
            'bracket_stats': self.bracket_stats,
            'institute_codes': self.institute_codes,
            'processed_candidates': self.processed_candidates
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Fitment scorer saved to {filepath}")
    
    def load_from_pkl(self, filepath: str = 'fitment_scorer.pkl'):
        """Load scorer state from pickle file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.config = data.get('config', self.config)
        self.brackets = data.get('brackets', self.brackets)
        self.bracket_stats = data.get('bracket_stats', {})
        self.institute_codes = data.get('institute_codes', {})
        self.processed_candidates = data.get('processed_candidates', {})
        print(f"✓ Fitment scorer loaded from {filepath}")
    
    def export_to_json(self, filepath: str = 'fitment_results.json'):
        """Export processed candidates to JSON for web use"""
        
        # FIX: Explicitly convert dictionary keys in bracket_stats to strings.
        safe_bracket_stats = {}
        for bracket, stats in self.bracket_stats.items():
            # Force the bracket number (key) to be a string
            safe_bracket_stats[str(bracket)] = stats
            
        export_data = {
            'config': self.config,
            'bracket_stats': safe_bracket_stats,  # Use the safe version
            'candidates': self.processed_candidates
        }
        with open(filepath, 'w') as f:
            # default=str handles NumPy float/int VALUES, while the loop handles KEYS
            json.dump(export_data, f, indent=2, default=str)
        print(f"✓ Results exported to {filepath}")
    
    def get_top_candidates(self, df: pd.DataFrame, n: int = 10, bracket: Optional[int] = None) -> pd.DataFrame:
        """Get top N candidates"""
        if bracket is not None:
            df_filtered = df[df['Experience_Bracket'] == bracket]
            print(f"\nTop {n} in Bracket {bracket} ({self.brackets[bracket]['label']}):")
        else:
            df_filtered = df
            print(f"\nTop {n} candidates:")
        
        display_cols = [col for col in [
            'Employee_Code', 'ID', 'Experience_Bracket', 'Category',
            'Technical_Score', 'Personality_Score', 'Fitment_Score',
            'Fitment_Percentile', 'Fitment_Category'
        ] if col in df_filtered.columns]
        
        return df_filtered[display_cols].nlargest(n, 'Fitment_Score')


class FitmentModelTrainer:
    """
    Production-ready Random Forest Model Trainer
    Trains on historically scored data for fitment prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.feature_importance = None
        self.training_stats = {}
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Fitment_Score') -> Tuple:
        """Prepare features for training"""
        df = df.copy()
        
        print(f"\n{'='*80}")
        print("PREPARING FEATURES FOR MODEL TRAINING")
        print(f"{'='*80}")
        
        numeric_features = [
            'Longevity_Years', 'Average_Experience', 'TotalPatents', 'TotalPapers',
            'Workshops', 'Trainings', 'Achievements', 'Books',
            'Number_of_Unique_Designations', 'State_J&K',
            'ExtroversionScore', 'AgreeablenessScore', 'ConscientiousnessScore',
            'NeuroticismScore', 'OpennessToExperienceScore'
        ]
        
        institute_features = ['UG_Institute', 'PG_Institute', 'PHD_Institute']
        categorical_features = ['Category', 'Experience_Bracket']
        
        available_numeric = [f for f in numeric_features if f in df.columns]
        available_institute = [f for f in institute_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        print(f"\nAvailable Features:")
        print(f"  Numeric: {len(available_numeric)}")
        print(f"  Institute: {len(available_institute)}")
        print(f"  Categorical: {len(available_categorical)}")
        
        for col in available_numeric + available_institute:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        for col in available_categorical:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_columns = (
            available_numeric + available_institute + 
            [f'{col}_encoded' for col in available_categorical]
        )
        
        print(f"\nTotal Features: {len(self.feature_columns)}")
        
        X = df[self.feature_columns].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        X = X.fillna(X.median())
        
        return X, y, df
    
    def train_model(self, df: pd.DataFrame, target_col: str = 'Fitment_Score',
                    test_size: float = 0.2, tune_hyperparameters: bool = False) -> Dict:
        """Train Random Forest model"""
        print(f"\n{'='*80}")
        print("TRAINING RANDOM FOREST MODEL")
        print(f"{'='*80}")
        
        X, y, df_processed = self.prepare_features(df, target_col)
        
        if y is None:
            raise ValueError(f"Target column '{target_col}' not found!")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=df_processed.get('Experience_Bracket', None)
        )
        
        print(f"\nData Split:")
        print(f"  Training: {len(X_train)}")
        print(f"  Testing: {len(X_test)}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if tune_hyperparameters:
            print("\nTuning hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, 25],
                'min_samples_split': [5, 10],
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best params: {grid_search.best_params_}")
        else:
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt',
                random_state=42, n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
        
        print(f"\n{'='*80}")
        print("MODEL EVALUATION")
        print(f"{'='*80}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")
        print(f"CV R² Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.training_stats = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'cv_mean_r2': float(cv_scores.mean()),
            'cv_std_r2': float(cv_scores.std()),
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'X_test': X_test, 'y_test': y_test, 'y_pred': y_test_pred,
            'X_train': X_train, 'y_train': y_train, 'y_train_pred': y_train_pred
        }
    
    def predict_new_candidates(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Predict fitment scores for new candidates"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        print(f"\nPredicting for {len(new_df)} candidates...")
        
        new_df = new_df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in new_df.columns:
                new_df[f'{col}_encoded'] = encoder.transform(new_df[col].astype(str))
        
        X_new = new_df[self.feature_columns].copy()
        X_new = X_new.fillna(X_new.median())
        X_new_scaled = self.scaler.transform(X_new)
        
        predictions = self.model.predict(X_new_scaled)
        tree_predictions = np.array([tree.predict(X_new_scaled) for tree in self.model.estimators_])
        
        new_df['Predicted_Fitment_Score'] = predictions
        new_df['Prediction_Std'] = tree_predictions.std(axis=0)
        new_df['Prediction_Lower'] = predictions - 1.96 * tree_predictions.std(axis=0)
        new_df['Prediction_Upper'] = predictions + 1.96 * tree_predictions.std(axis=0)
        
        print(f"✓ Predictions complete!")
        print(f"  Mean: {predictions.mean():.2f}")
        print(f"  Range: {predictions.min():.2f} - {predictions.max():.2f}")
        
        return new_df
    
    def save_model(self, model_dir: str = '.'):
        """Save trained model and preprocessing objects"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        Path(model_dir).mkdir(exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/fitment_model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/fitment_scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_dir}/fitment_encoders.pkl')
        
        metadata = {
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats,
            'feature_importance': self.feature_importance.to_dict('records')
        }
        joblib.dump(metadata, f'{model_dir}/fitment_metadata.pkl')
        
        print(f"\n✅ Model saved successfully!")
        print(f"  Model: {model_dir}/fitment_model.pkl")
        print(f"  Scaler: {model_dir}/fitment_scaler.pkl")
        print(f"  Encoders: {model_dir}/fitment_encoders.pkl")
        print(f"  Metadata: {model_dir}/fitment_metadata.pkl")
    
    def load_model(self, model_dir: str = '.'):
        """Load trained model and preprocessing objects"""
        self.model = joblib.load(f'{model_dir}/fitment_model.pkl')
        self.scaler = joblib.load(f'{model_dir}/fitment_scaler.pkl')
        self.label_encoders = joblib.load(f'{model_dir}/fitment_encoders.pkl')
        
        metadata = joblib.load(f'{model_dir}/fitment_metadata.pkl')
        self.feature_columns = metadata['feature_columns']
        self.training_stats = metadata['training_stats']
        self.feature_importance = pd.DataFrame(metadata['feature_importance'])
        
        print(f"\n✅ Model loaded successfully!")
        print(f"  R² Score: {self.training_stats['test_r2']:.4f}")
        print(f"  RMSE: {self.training_stats['test_rmse']:.4f}")
    
    def visualize_model_performance(self, results: Dict):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6, s=50)
        axes[0, 0].plot([results['y_test'].min(), results['y_test'].max()],
                        [results['y_test'].min(), results['y_test'].max()],
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Fitment Score')
        axes[0, 0].set_ylabel('Predicted Fitment Score')
        axes[0, 0].set_title('Actual vs Predicted (Test Set)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        residuals = results['y_test'] - results['y_pred']
        axes[0, 1].scatter(results['y_pred'], residuals, alpha=0.6, s=50)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Fitment Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].hist(residuals, bins=30, color='skyblue', edgecolor='black')
        axes[0, 2].set_xlabel('Prediction Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Error Distribution', fontweight='bold')
        axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
        
        top_features = self.feature_importance.head(15)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'], fontsize=9)
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importances', fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        metrics = ['R² Score', 'RMSE', 'MAE']
        train_vals = [self.training_stats['train_r2'], 0, 0]
        test_vals = [self.training_stats['test_r2'],
                    self.training_stats['test_rmse'],
                    self.training_stats['test_mae']]
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[1, 1].bar(x - width/2, train_vals, width, label='Training', color='skyblue')
        axes[1, 1].bar(x + width/2, test_vals, width, label='Testing', color='coral')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Training vs Testing Performance', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        axes[1, 2].scatter(results['y_test'], np.abs(residuals), alpha=0.6, s=50)
        axes[1, 2].set_xlabel('Actual Fitment Score')
        axes[1, 2].set_ylabel('Absolute Error')
        axes[1, 2].set_title('Prediction Confidence', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# COMBINED PRODUCTION PIPELINE
# ============================================================================

class FitmentScoringPipeline:
    """
    Complete integrated pipeline for fitment scoring and model training
    Handles end-to-end workflow with pkl serialization for website integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.scorer = EnhancedFitmentScoringSystem(config=config)
        self.trainer = FitmentModelTrainer()
        self.pipeline_results = {}
    
    def run_complete_pipeline(self, df: pd.DataFrame, model_dir: str = '.',
                             tune_hyperparameters: bool = False):
        """
        Run complete pipeline: scoring → training → serialization
        """
        print(f"\n{'='*80}")
        print("COMPLETE FITMENT SCORING & TRAINING PIPELINE")
        print(f"{'='*80}")
        
        # Step 1: Score all candidates
        print("\n[1/4] Scoring candidates...")
        scored_df = self.scorer.process_fitment_scoring(df)
        
        # Step 2: Train model
        print("\n[2/4] Training model...")
        training_results = self.trainer.train_model(
            scored_df,
            target_col='Fitment_Score',
            test_size=0.2,
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Step 3: Save models and scorers
        print("\n[3/4] Serializing to pkl files...")
        self.scorer.save_to_pkl(f'{model_dir}/fitment_scorer.pkl')
        self.trainer.save_model(model_dir)
        self.scorer.export_to_json(f'{model_dir}/fitment_results.json')
        
        # Step 4: Summary
        print("\n[4/4] Pipeline complete!")
        self.pipeline_results = {
            'scored_candidates': len(scored_df),
            'model_performance': self.trainer.training_stats,
            'scorer_path': f'{model_dir}/fitment_scorer.pkl',
            'model_path': f'{model_dir}/fitment_model.pkl'
        }
        
        return scored_df, training_results
    
    def load_pipeline(self, model_dir: str = '.'):
        """Load saved scorer and model for inference"""
        self.scorer.load_from_pkl(f'{model_dir}/fitment_scorer.pkl')
        self.trainer.load_model(model_dir)
        print("\n✓ Pipeline loaded successfully!")
    
    import pandas as pd
    from typing import Dict

# Note: This implementation assumes the full set of required features 
# (including all personality, institute flags, etc.) are present in candidate_data.

    def score_single_candidate(self, candidate_data: Dict) -> Dict:
        """
        Score a single candidate using the loaded rule-based scorer.
        It simulates the full pipeline logic using loaded bracket_stats for accurate, 
        bracket-relative scoring.
        """
        if not self.scorer.bracket_stats:
            raise ValueError("Scorer not initialized! Load pipeline first.")
    
        # 1. Convert the single candidate dictionary to a DataFrame
        #    This is necessary because the scoring functions (calculate_*) operate on DataFrames.
        df_single = pd.DataFrame([candidate_data])
        
        # 2. Add necessary classification columns (Experience_Bracket and Category)
        df_single['Experience_Bracket'] = df_single['Longevity_Years'].apply(self.scorer.assign_experience_bracket)
        
        # We define get_category here to avoid relying on the full load_data process
        def get_category_single(row):
            if row['Longevity_Years'] < 2:
                return 'Fresher'
            elif row['Longevity_Years'] < 5:
                return 'Inexperienced'
            else:
                return 'Experienced'
                
        df_single['Category'] = df_single.apply(get_category_single, axis=1)
    
        # 3. Calculate Technical Score (uses loaded self.scorer.bracket_stats for relative scoring)
        df_single = self.scorer.calculate_technical_score(df_single)
        
        # 4. Calculate Personality Score (performs bracket-relative normalization)
        df_single = self.scorer.calculate_personality_score(df_single)
    
        # 5. Calculate Final Fitment Score 
        df_single = self.scorer.calculate_fitment_score(df_single)
    
        # 6. Extract the complete results for return
        result = df_single.iloc[0].to_dict()
    
        return {
            'fitment_score': round(result.get('Fitment_Score', 0), 2),
            'fitment_category': result.get('Fitment_Category'),
            'bracket': result.get('Experience_Bracket'),
            'category': result.get('Category'),
            'personality_score': round(result.get('Personality_Score', 0), 2),
            'technical_score': round(result.get('Technical_Score', 0), 2)
        }    
    def predict_candidate(self, candidate_data: Dict) -> Dict:
        """Predict fitment using trained ML model"""
        if self.trainer.model is None:
            raise ValueError("Model not loaded! Load pipeline first.")
        
        df = pd.DataFrame([candidate_data])
        predictions = self.trainer.predict_new_candidates(df)
        
        result = predictions.iloc[0].to_dict()
        result['predicted_fitment_score'] = round(result.get('Predicted_Fitment_Score', 0), 2)
        result['prediction_confidence'] = round(100 - result.get('Prediction_Std', 0) * 10, 2)
        
        return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load your dataset
    print("Loading dataset...")
    df = pd.read_excel('Final_Dataset.xlsx')  # or .xlsx
    
    # Initialize and run complete pipeline
    pipeline = FitmentScoringPipeline(config={
        'use_tiered_scoring': True,
        'use_bracket_relative': True,
        'include_institute_bonus': True,
        'include_state_bonus': True,
        'personality_weight_fresher': 0.70,
        'technical_weight_fresher': 0.30,
        'personality_weight_experienced': 0.30,
        'technical_weight_experienced': 0.70,
    })
    
    # Run complete pipeline
    scored_df, training_results = pipeline.run_complete_pipeline(
        df,
        model_dir='./fitment_models',
        tune_hyperparameters=False
    )
    
    # Visualize results
    pipeline.trainer.visualize_model_performance(training_results)
    
    # Save scored results
    scored_df.to_csv('fitment_scored_results.csv', index=False)
    print("\n✅ Results saved to fitment_scored_results.csv")
    
    # ========================================================================
    # EXAMPLE: USING SAVED MODELS FOR INFERENCE
    # ========================================================================
    print(f"\n{'='*80}")
    print("INFERENCE EXAMPLE - SCORING NEW CANDIDATES")
    print(f"{'='*80}")
    
    # Load pipeline for inference
    inference_pipeline = FitmentScoringPipeline()
    inference_pipeline.load_pipeline('./fitment_models')
    
    # Score new candidate using rule-based scorer
    new_candidate_1 = {
        # --- 6 Features Currently Present ---
        'Longevity_Years': 3.5,
        'Average_Experience': 3.0,
        'TotalPatents': 2,
        'TotalPapers': 1,
        'Workshops': 3,
        'Trainings': 2,
    
        # --- 14 Missing Features (MUST BE ADDED) ---
        
        # Missing Technical/Experience
        'Achievements': 1,  # Added
        'Books': 0,         # Added
        'Number_of_Unique_Designations': 3, # Added (must be estimated)
        
        # Missing Demographic/Institute Flags (Set to 0/False unless known)
        'State_J&K': 0,     # Added
        'UG_Institute': 0,  # Added (Model was trained on 0/1 indicator flags)
        'PG_Institute': 0,  # Added
        'PHD_Institute': 0, # Added
        
        # Missing Personality Scores (Use plausible values, e.g., 30/40)
        'ExtroversionScore': 30,         # Added
        'AgreeablenessScore': 35,          # Added
        'ConscientiousnessScore': 38,    # Added
        'NeuroticismScore': 10,          # Added (Lower score = Better)
        'OpennessToExperienceScore': 32, # Added
        
        # Base Categorical Columns (Required for the LabelEncoder)
        'Experience_Bracket': 2,  # Added (Since Longevity_Years is 3.5, bracket is 2)
        'Category': 'Experienced' # Added (Since Longevity_Years is 3.5, category is Experienced)
}
    
    score_result = inference_pipeline.score_single_candidate(new_candidate_1)
    print(f"\nRule-based Score: {score_result}")
    
    # Predict using ML model
    prediction_result = inference_pipeline.predict_candidate(new_candidate_1)
    print(f"\nML Model Prediction: {prediction_result}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE PIPELINE EXECUTION FINISHED!")
    print(f"{'='*80}")