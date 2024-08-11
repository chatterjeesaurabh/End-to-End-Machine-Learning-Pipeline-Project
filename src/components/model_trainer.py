import os
import sys
from src.logger.logging import logging
from src.exception.exception import customException
from src.utils.utils import evaluate_model, save_object

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge


@dataclass
class ModelTrainerConfig:
    pass


class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customException(e, sys)



