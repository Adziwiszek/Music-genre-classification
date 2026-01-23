from sklearn import svm
from sklearn.svm import LinearSVC
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
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
    csv_path: str
    pipeline : Pipeline
    drop_cols : list 
    target_col: str = 'label'
    test_size: float = 0.3
    
@dataclass
class EvalModel():
    config: EvalConfig
    y_test: Any = field(default=None, init=False)
    y_pred: Any = field(default=None, init=False)
    
    def get_feature_scores(self, X):
        """Selects k best if SelectKBest is in Pipeline."""
        if 'select_best' in self.config.pipeline.named_steps:
            selector = self.config.pipeline.named_steps['select_best']
            feature_names = X.columns
            scores = selector.scores_
            
            df_scores = pd.DataFrame({'Feature': feature_names, 'Score': scores})
            return df_scores.sort_values(by='Score', ascending=False)
        return "'select_best' not in Pipeline"

    def evaluate(self):
        df = pd.read_csv(self.config.csv_path)
        X, y = df.drop(self.config.drop_cols, axis=1, errors='ignore'), df[self.config.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=32,
            stratify=y
        )
        
        self.config.pipeline.fit(X_train, y_train)
        
        self.y_test = y_test
        self.y_pred = self.config.pipeline.predict(X_test)
        
        feature_importance = self.get_feature_scores(X_test)
        
        return {
            "report": classification_report(self.y_test, self.y_pred, output_dict=True),
            "feature_importance": feature_importance
        }

    def plot_eval(self):
        results = self.evaluate()

        report_dict = results["report"]
        accuracy_val = report_dict["accuracy"]
    
        report_df = pd.DataFrame(report_dict).T
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        report_df['accuracy'] = accuracy_val
        report_df = report_df[['precision', 'recall', 'f1-score', 'accuracy']]
    
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(1, 3, figsize=(22, 7))

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
    
        # Feature Importance (SelectKBest)
        feat_imp = results['feature_importance']
        if isinstance(feat_imp, pd.DataFrame):
            sns.barplot(data=feat_imp.head(10), x='Score', y='Feature', hue='Feature', palette='viridis', ax=ax[2])
            ax[2].set_title('Top 10 Features (SelectKBest Scores)')
        else:
            ax[2].text(0.5, 0.5, "SelectKBest not found in Pipeline", ha='center')
    
        plt.tight_layout()
        plt.show()