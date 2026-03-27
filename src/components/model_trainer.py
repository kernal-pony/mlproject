from dataclasses import dataclass
import os
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # 🔥 USE STRONG MODEL ONLY
            model = RandomForestRegressor(n_estimators=100)

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            print(f"🔥 Model R2 Score: {score}")

            save_object(self.config.trained_model_file_path, model)

            return score

        except Exception as e:
            raise CustomException(e, sys)