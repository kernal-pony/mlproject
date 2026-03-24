import pandas as pd

class DataTransformationConfig:
    pass


class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_transformation(self, train_path, test_path):
        print("Data transformation started...")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Dummy transformation
        train_arr = train_df.values
        test_arr = test_df.values

        return train_arr, test_arr, None