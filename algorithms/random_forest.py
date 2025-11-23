from sklearn.ensemble import RandomForestClassifier
from algorithms.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, random_state=42, max_depth=None):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth
        )
        super().__init__("Random Forest", model)
    
    def get_feature_importance(self, feature_names):
        """Retorna a importância das features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
        return None
