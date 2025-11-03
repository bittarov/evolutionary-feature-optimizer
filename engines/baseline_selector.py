"""
Baseline Feature Selection Methods
Traditional statistical and machine learning approaches for comparison
"""
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class BaselineSelector:
    """
    Collection of traditional feature selection methods for benchmarking
    Provides statistical, information-theoretic, and model-based approaches
    """
    
    def __init__(self, X, y):
        """
        Initialize baseline selector
        
        Args:
            X: Feature matrix (numpy array)
            y: Target vector (numpy array)
        """
        self.X = X
        self.y = y
    
    def statistical_test(self, k=None):
        """
        Select features using ANOVA F-test statistical significance
        Measures linear dependency between features and target
        
        Args:
            k: Number of top features to select
            
        Returns:
            Tuple: (selected_feature_indices, test_scores)
        """
        if k is None:
            k = min(50, self.X.shape[1] // 2)
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, selector.scores_
    
    def information_theory(self, k=None):
        """
        Select features using Mutual Information criterion
        Measures information gain and non-linear dependencies
        
        Args:
            k: Number of top features to select
            
        Returns:
            Tuple: (selected_feature_indices, information_scores)
        """
        if k is None:
            k = min(50, self.X.shape[1] // 2)
        selector = SelectKBest(mutual_info_classif, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, selector.scores_
    
    def recursive_elimination(self, n_features=None):
        """
        Recursive Feature Elimination using Random Forest
        Iteratively removes least important features
        
        Args:
            n_features: Target number of features
            
        Returns:
            Tuple: (selected_feature_indices, feature_rankings)
        """
        if n_features is None:
            n_features = min(50, self.X.shape[1] // 2)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(model, n_features_to_select=n_features)
        selector.fit(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, selector.ranking_
    
    def tree_based_importance(self, threshold='median'):
        """
        Select features based on Random Forest feature importance
        Uses Gini importance from decision trees
        
        Args:
            threshold: Importance threshold for selection
            
        Returns:
            Tuple: (selected_feature_indices, importance_scores)
        """
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)
        selector = SelectFromModel(model, threshold=threshold)
        selector.fit(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, model.feature_importances_
    
    def compute_accuracy(self, selected_features):
        """
        Evaluate accuracy of selected feature subset
        Uses cross-validated Random Forest classifier
        
        Args:
            selected_features: List of feature indices
            
        Returns:
            Mean cross-validation accuracy
        """
        if len(selected_features) == 0:
            return 0
        X_selected = self.X[:, selected_features]
        # Single-threaded execution to avoid multiprocessing issues
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        scores = cross_val_score(model, X_selected, self.y, cv=5, scoring='accuracy', n_jobs=1)
        return scores.mean()
    
    def compare_all(self, evolutionary_features, evolutionary_score):
        """
        Compare all baseline methods against evolutionary approach
        Provides comprehensive benchmarking
        
        Args:
            evolutionary_features: Features selected by evolutionary optimizer
            evolutionary_score: Accuracy score from evolutionary method
            
        Returns:
            Dictionary with results from all methods
        """
        results = {
            'evolutionary': {
                'features': evolutionary_features,
                'feature_count': len(evolutionary_features),
                'accuracy': evolutionary_score
            }
        }
        
        # Statistical test (F-test)
        stat_features, stat_scores = self.statistical_test(len(evolutionary_features))
        results['statistical'] = {
            'features': stat_features,
            'feature_count': len(stat_features),
            'accuracy': self.compute_accuracy(stat_features),
            'scores': stat_scores
        }
        
        # Information theory (Mutual Information)
        info_features, info_scores = self.information_theory(len(evolutionary_features))
        results['information'] = {
            'features': info_features,
            'feature_count': len(info_features),
            'accuracy': self.compute_accuracy(info_features),
            'scores': info_scores
        }
        
        # Recursive elimination
        rec_features, rec_ranking = self.recursive_elimination(len(evolutionary_features))
        results['recursive'] = {
            'features': rec_features,
            'feature_count': len(rec_features),
            'accuracy': self.compute_accuracy(rec_features)
        }
        
        # Tree-based importance
        tree_features, tree_importances = self.tree_based_importance()
        results['tree_based'] = {
            'features': tree_features,
            'feature_count': len(tree_features),
            'accuracy': self.compute_accuracy(tree_features),
            'importances': tree_importances
        }
        
        return results

