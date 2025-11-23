from sklearn.tree import DecisionTreeClassifier
from algorithms.base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self, random_state=42, max_depth=None, min_samples_split=2):
        model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        super().__init__("Decision Tree", model)
    
    def get_feature_importance(self, feature_names):
        """Retorna a importância das features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
        return None
