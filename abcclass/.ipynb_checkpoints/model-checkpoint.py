import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Optional

class Model(metaclass=ABCMeta):
    
    """Abstract Class for ML Model
    
    Abstract Methods
    -----------
    train : training and saving your model
    predict : your model predicts using test data and return prediction
    save_model : saving your model
    load_model : loading your model
    tune(unimplemented) : tuning your model
    """
    
    def __init__(self, model_name: str, params: dict) -> None:
        
        """constructor
        
        argument
        -----------
        model_name(str) : name of your model(this argument will be used file name saving model)
        params(dict) : your model has params
        model(sklearn.estimator etc) : valiable saving your model
        """
        self.model_name = model_name
        self.params = params
        self.model = None
        
    @abstractmethod    
    def train(self, train_X: pd.DataFrame, train_y: pd.Series, 
              val_X: Optional[pd.DataFrame]=None,  val_y: Optional[pd.DataFrame]=None) -> None:
        
        """training and saving your model
        
        argument
        -----------
        train_X(pandas.DataFrame) : features of training data
        train_y(pandas.Series): objective variable of training data
        val_X(pandas.DataFrame) : features of validation data
        val_y(pandas.Series) : objective variable of validation data
        """
        
        pass
    
    
    @abstractmethod
    def predict(self, test_X: pd.DataFrame) -> np.ndarray:
        
        """your model predicts using test data and return prediction
        
        argument
        -----------
        test_X(pandas.DataFrame) : features of test data
        
        return
        -----------
        pred(numpy.ndarray) : prediction result
        """
        
        pass
    
    
    @abstractmethod
    def tune(self, tune_params: dict) -> dict:
        """tuning your model(use GridSearch, Optuna etc)
        
        argument
        -----------
        tune_params(dict) : model parameters you want to tune
        
        return
        -----------
        best_params(dict) : parameters result is best score
        """
        
        pass
    
    
    @abstractmethod
    def save_model(self) -> None:
        
        """saving your model"""
        
        pass
    
    
    @abstractmethod
    def load_model(self) -> None:
        
        """loading your model"""
        
        pass
    
    