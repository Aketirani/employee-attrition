import unittest
from unittest.mock import mock_open, patch

import pandas as pd

from employeeattrition import EmployeeAttrition


class TestEmployeeAttrition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg_file = "config/config.yaml"
        cls.ea = EmployeeAttrition(cls.cfg_file)

    def test_read_config(self):
        cfg_data = self.ea.read_config(self.cfg_file)
        self.assertIsInstance(cfg_data, dict)

    def test_load_data(self):
        data = self.ea.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_feature_engineering(self):
        data = pd.DataFrame(
            {
                "average_monthly_hours": [160, 150, 140],
                "number_project": [3, 4, 5],
                "left": [0, 1, 0],
            }
        )
        result = self.ea.feature_engineering(data)
        self.assertIn("interaction_feature", result.columns)
        self.assertIn("projects_per_hour", result.columns)
        self.assertIn("avg_hours_per_project", result.columns)

    def test_save_data_description(self):
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "target": [0, 1, 0]})
        with patch("builtins.open", mock_open()) as mock_file:
            self.ea.save_data_description(data)
            mock_file().write.assert_called()

    def test_plot_average_monthly_hours_vs_left(self):
        data = pd.DataFrame(
            {"average_monthly_hours": [160, 150, 140], "left": [0, 1, 0]}
        )
        with patch("seaborn.jointplot"), patch("matplotlib.pyplot.savefig"):
            self.ea.plot_average_monthly_hours_vs_left(data)

    def test_calculate_pearson_correlation(self):
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "left": [0, 1, 0]})
        corr_matrix = self.ea.calculate_pearson_correlation(data)
        self.assertIsInstance(corr_matrix, pd.DataFrame)

    def test_drop_columns(self):
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
        columns_to_drop = ["col2", "col3"]
        result = self.ea.drop_columns(data, columns_to_drop)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertNotIn("col2", result.columns)
        self.assertNotIn("col3", result.columns)
        self.assertIn("col1", result.columns)

    def test_one_hot_encode(self):
        data = pd.DataFrame(
            {
                "department": ["HR", "IT", "HR"],
                "salary": ["low", "high", "medium"],
                "left": [0, 1, 0],
            }
        )
        columns_to_encode = ["salary"]
        result = self.ea.one_hot_encode(data, columns_to_encode)
        self.assertNotIn("salary", result.columns)

    def test_split_data(self):
        data = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "col2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "left": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        train_data, val_data, test_data = self.ea.split_data(data)
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(val_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)

    def test_prepare_data(self):
        train_data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "left": [0, 1, 0]}
        )
        val_data = pd.DataFrame(
            {"col1": [4, 5, 6], "col2": [7, 8, 9], "left": [1, 0, 1]}
        )
        test_data = pd.DataFrame(
            {"col1": [7, 8, 9], "col2": [10, 11, 12], "left": [0, 1, 0]}
        )
        train_X, train_y, val_X, val_y, test_X, test_y = self.ea.prepare_data(
            train_data, val_data, test_data
        )
        self.assertIsInstance(train_X, pd.DataFrame)
        self.assertIsInstance(train_y, pd.Series)
        self.assertIsInstance(val_X, pd.DataFrame)
        self.assertIsInstance(val_y, pd.Series)
        self.assertIsInstance(test_X, pd.DataFrame)
        self.assertIsInstance(test_y, pd.Series)

    def test__load_hyperparameters(self):
        hyperparameters = self.ea._load_hyperparameters()
        self.assertIsInstance(hyperparameters, dict)

    def test__load_model_parameters(self):
        model_parameters = self.ea._load_model_parameters()
        self.assertIsInstance(model_parameters, dict)

    @patch("xgboost.XGBClassifier.predict")
    def test_calculate_accuracy(self, mock_predict):
        test_y = pd.Series([0, 1, 0])
        predicted_values = [0, 1, 0]
        accuracy = self.ea.calculate_accuracy(test_y, predicted_values)
        self.assertIsInstance(accuracy, float)

    @patch("pandas.DataFrame.to_excel")
    @patch("builtins.open", mock_open())
    def test_save_predicted_output(self, mock_to_excel):
        test_X = pd.DataFrame(
            {
                "col1": [7, 8, 9],
                "col2": [10, 11, 12],
            }
        )
        predicted_values = [0, 1, 0]
        test_y = pd.Series([0, 1, 0])
        self.ea.save_predicted_output(test_X, predicted_values, test_y)
        mock_to_excel.assert_called()
