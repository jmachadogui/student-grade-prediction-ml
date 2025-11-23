from sklearn.naive_bayes import GaussianNB
from algorithms.base_model import BaseModel

class NaiveBayesModel(BaseModel):
    def __init__(self, var_smoothing=1e-9):
        model = GaussianNB(var_smoothing=var_smoothing)
        super().__init__("Naive Bayes", model)
    
    def get_class_priors(self):
        """Retorna as probabilidades a priori das classes"""
        if hasattr(self.model, 'class_prior_'):
            return self.model.class_prior_
        return None
    
    def get_theta(self):
        """Retorna as médias de cada feature por classe"""
        if hasattr(self.model, 'theta_'):
            return self.model.theta_
        return None
