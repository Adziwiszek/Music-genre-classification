from sklearn import svm
from sklearn.svm import LinearSVC
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

@dataclass
class EvalConfig:
    train_path: str               # or full path
    pipeline : Pipeline
    drop_cols : list 

    test_path: Optional[str] = None
    target_col: str = 'label'
    test_size: float = 0.2
    random_state: int = 42
    
@dataclass
class EvalModel():
    config: EvalConfig
    y_test: Any = field(default=None, init=False)
    y_pred: Any = field(default=None, init=False)
    
    def get_feature_scores(self, X):
        """
        Selects k best if SelectKBest is in Pipeline.
        If not, we show coef_ or feature_importances_, depending on the model
        """
        if not hasattr(self.config.pipeline, 'named_steps'):
            self.feature_source_name = "N/A (Ensemble/Voting)"
        return None

        feature_names = X.columns
        importances = None
        self.feature_source_name = "N/A"
        if 'select_best' in self.config.pipeline.named_steps:
            selector = self.config.pipeline.named_steps['select_best']
            importances = selector.scores_
            self.feature_source_name = "SelectKBest Scores"
        else:
            # last step in pipline
            estimator = estimator = self.config.pipeline.steps[-1][1]

            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                self.feature_source_name = f"Feature Importance ({estimator.__class__.__name__})"
           
            elif hasattr(estimator, 'coef_'):
                # coef_ has (n_classes, n_features) or (1, n_features)
                if estimator.coef_.ndim > 1:
                    importances = np.mean(np.abs(estimator.coef_), axis=0)
                else:
                    importances = np.abs(estimator.coef_).flatten()
                self.feature_source_name = f"Abs Coefficients ({estimator.__class__.__name__})"
                
        if importances is not None:
            if len(importances) == len(feature_names):
                df_scores = pd.DataFrame({'Feature': feature_names, 'Score': importances})
                return df_scores.sort_values(by='Score', ascending=False)
        
        return None

    def _prepare_data(self, filepath: str):
        df = pd.read_csv(filepath)
        X, y = df.drop(self.config.drop_cols, axis=1, errors='ignore'), df[self.config.target_col]
        return X, y

    def evaluate(self):
        if self.config.test_path:
            X_train, y_train = self._prepare_data(self.config.train_path)
            X_test, y_test = self._prepare_data(self.config.test_path)
        
        else:
            X, y = self._prepare_data(self.config.train_path)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
        
        self.config.pipeline.fit(X_train, y_train)
        
        self.y_test = y_test
        self.y_pred = self.config.pipeline.predict(X_test)
        
        feature_importance = self.get_feature_scores(X_test)
        
        return {
            "report": classification_report(self.y_test, self.y_pred, output_dict=True,zero_division=0),
            "feature_importance": feature_importance
        }

    def plot_eval(self):
        results = self.evaluate()

        report_dict = results["report"]
        accuracy_val = report_dict.get("accuracy", 0)
    
        report_df = pd.DataFrame(report_dict).T
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        report_df = report_df[['precision', 'recall', 'f1-score']]

        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(1, 3, figsize=(22, 7))
        plt.suptitle(f"Model Evaluation Dashboard (Overall Accuracy: {accuracy_val:.2%})", fontsize=16, weight='bold')

        # Metrics (Heatmap)
        sns.heatmap(report_df, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax[0])
        ax[0].set_title('Classification Metrics per Genre')
    
        # Confusion matrix 
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1],
                    xticklabels=self.config.pipeline.classes_, 
                    yticklabels=self.config.pipeline.classes_)
        ax[1].set_title('Confusion Matrix - Music Genres')
        ax[1].set_xlabel('Predicted Label')
        ax[1].set_ylabel('True Label')
    
        # Feature Importance
        feat_imp = results['feature_importance']
        if isinstance(feat_imp, pd.DataFrame):
            sns.barplot(data=feat_imp.head(10), x='Score', y='Feature', hue='Feature', palette='viridis', ax=ax[2])
            ax[2].set_title(f'Top 10 Features \nSource: {self.feature_source_name}')
            ax[2].set_xlabel('Importance Score')
        else:
            ax[2].text(0.5, 0.5, "None", ha='center')
    
        plt.tight_layout()
        plt.show()