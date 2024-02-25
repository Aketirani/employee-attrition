import argparse
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


class EmployeeAttrition:
    def __init__(self, cfg_file, model_type):
        """
        Initialize the class.

        :param cfg_file: str, path to the configuration file.
        :param model_type: str, type of model to run.
        :attr config_file: dict, dictionary containing configuration parameters.
        :attr paths: dict, dictionary containing all the file paths.
        :attr column_names: dict, dictionary containing column names.
        """
        self.config_file = self.read_config(cfg_file)
        self.model_type = model_type
        self.paths = {
            "data_raw": self.config_file.get("data", {}).get("raw"),
            "data_pred": self.config_file.get("data", {}).get("pred"),
            "data_description": self.config_file.get("results", {}).get(
                "data_description"
            ),
            "distribution_columns": self.config_file.get("plots", {}).get(
                "distribution_columns"
            ),
            "average_monthly_hours": self.config_file.get("plots", {}).get(
                "average_monthly_hours"
            ),
            "correlation_matrix": self.config_file.get("plots", {}).get(
                "correlation_matrix"
            ),
            "model_hyperparameters": self.config_file.get("parameters", {}).get(
                "model_hyperparameters"
            ),
            "model_best_param": self.config_file.get("results", {}).get(
                "model_best_param"
            ),
            "model_parameters": self.config_file.get("parameters", {}).get(
                "model_parameters"
            ),
            "model_object": self.config_file.get("results", {}).get("model_object"),
            "confusion_matrix": self.config_file.get("plots", {}).get(
                "confusion_matrix"
            ),
            "model_performance": self.config_file.get("plots", {}).get(
                "model_performance"
            ),
            "feature_importance": self.config_file.get("plots", {}).get(
                "feature_importance"
            ),
            "shapley_summary": self.config_file.get("plots", {}).get("shapley_summary"),
        }
        self.column_names = {
            "to_drop": self.config_file.get("columns", {}).get("drop"),
            "to_encode": self.config_file.get("columns", {}).get("encode"),
            "target": self.config_file.get("columns", {}).get("target"),
            "predicted": self.config_file.get("columns", {}).get("predicted"),
            "difference": self.config_file.get("columns", {}).get("difference"),
        }

    def read_config(self, cfg_file) -> dict:
        """
        Read the config yaml file and return the data as a dictionary

        :param cfg_file: str, path to the configuration file
        :return: dict, containing the configuration data
        """
        try:
            with open(cfg_file, "r") as file:
                cfg_setup = yaml.safe_load(file)
        except:
            raise FileNotFoundError(f"{cfg_file} is not a valid config filepath!")
        return cfg_setup

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the specified raw data file.

        :return: pd.DataFrame, loaded dataset
        """
        return pd.read_excel(self.paths["data_raw"])

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the given DataFrame.

        :param df: pd.DataFrame, input dataset
        :return: pd.DataFrame, dataset after feature engineering
        """
        df["interaction_feature"] = df["average_monthly_hours"] * df["number_project"]
        df["projects_per_hour"] = df["number_project"] / df["average_monthly_hours"]
        df["avg_hours_per_project"] = df["average_monthly_hours"] / df["number_project"]
        return df

    def save_data_description(self, df: pd.DataFrame) -> None:
        """
        Save data description to a text file.

        :param df: pd.DataFrame, input dataset
        """
        output_string = f"Shape of the dataset: {df.shape}\n\n"
        output_string += f"Column Descriptions:\n{df.describe().to_markdown()}\n\n"
        missing_values_summary = pd.DataFrame(
            {"Column": df.columns, "Missing Values": df.isnull().sum()}
        ).reset_index(drop=True)
        output_string += (
            f"Missing Values:\n{missing_values_summary.to_markdown(index=False)}\n"
        )
        with open(self.paths["data_description"], "w") as file:
            file.write(output_string)

    def plot_distribution_columns(self, df: pd.DataFrame) -> None:
        """
        Plot the distribution of each column in the DataFrame.

        :param df: pd.DataFrame, input dataset
        """
        columns_per_row = 3
        num_rows = int(np.ceil(len(df.columns) / columns_per_row))
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=columns_per_row, figsize=(15, 5 * num_rows)
        )
        fig.subplots_adjust(hspace=0.5)

        for i, column in enumerate(df.columns):
            row_index = i // columns_per_row
            col_index = i % columns_per_row
            ax = axes[row_index, col_index]
            sns.histplot(
                x=column,
                data=df,
                hue="left",
                multiple="stack",
                bins=20,
                palette="pastel",
                edgecolor="black",
                ax=ax,
            )
            ax.set_title(f"Distribution of {column}")
            ax.tick_params(axis="x", rotation=45)

        for i in range(len(df.columns), num_rows * columns_per_row):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.savefig(self.paths["distribution_columns"])
        plt.close()

    def plot_average_monthly_hours_vs_left(self, df: pd.DataFrame) -> None:
        """
        Plot average monthly hours vs left.

        :param df: pd.DataFrame, input dataset
        """
        sns.jointplot(
            x="average_monthly_hours",
            y=self.column_names["target"],
            data=df,
            kind="kde",
            color="skyblue",
            fill=True,
        )
        plt.title("Average Monthly Hours vs Left")
        plt.yticks([0, 1])
        plt.tight_layout()
        plt.savefig(self.paths["average_monthly_hours"])
        plt.close()

    def calculate_pearson_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Pearson correlation matrix excluding the target column.

        :param df: pd.DataFrame, input dataset
        :return: pd.DataFrame, Pearson correlation matrix
        """
        df_features = df.drop(self.column_names["target"], axis=1)
        return df_features.corr(numeric_only=True)

    def plot_pearson_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """
        Plot the Pearson correlation matrix.

        :param corr_matrix: pd.DataFrame, Pearson correlation matrix
        """
        plt.figure(figsize=(8, 8))
        heatmap = sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right")
        plt.title("Pearson Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(self.paths["correlation_matrix"])
        plt.close()

    def drop_columns(self, df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.

        :param df: pd.DataFrame, input dataset
        :param columns_to_drop: list, names of the columns to be dropped
        :return: pd.DataFrame, dataset after dropping the specified columns
        """
        return df.drop(columns=columns_to_drop, axis=1, errors="ignore")

    def one_hot_encode(self, df: pd.DataFrame, columns_to_encode: list) -> pd.DataFrame:
        """
        Perform one-hot encoding on specified categorical columns.

        :param df: pd.DataFrame, input dataset
        :param columns_to_encode: list, names of the columns to be one-hot encoded
        :return: pd.DataFrame, dataset after one-hot encoding
        """
        return pd.get_dummies(df, columns=columns_to_encode, drop_first=False)

    def split_data(
        self, df: pd.DataFrame, train_size: float = 0.8, test_size: float = 0.5
    ) -> tuple:
        """
        Split the dataset into training, validation, and testing sets.

        :param df: pd.DataFrame, input dataset
        :param train_size: float, training set size for train-test split (default is 0.8)
        :param test_size: float, test size for validation and test split (default is 0.5)
        :return: tuple, containing training, validation, and testing datasets
        """
        train_data, other_data = train_test_split(
            df, test_size=(1 - train_size), random_state=42
        )
        val_data, test_data = train_test_split(
            other_data, test_size=test_size, random_state=42
        )
        return train_data, val_data, test_data

    def prepare_data(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> tuple:
        """
        Prepare the training, validation, and testing data.

        :param train_data: pd.DataFrame, training dataset
        :param val_data: pd.DataFrame, validation dataset
        :param test_data: pd.DataFrame, testing dataset
        :return: tuple, containing training, validation and testing features and labels
        """
        train_X = train_data.drop(self.column_names["target"], axis=1)
        train_y = train_data[self.column_names["target"]]
        val_X = val_data.drop(self.column_names["target"], axis=1)
        val_y = val_data[self.column_names["target"]]
        test_X = test_data.drop(self.column_names["target"], axis=1)
        test_y = test_data[self.column_names["target"]]
        return train_X, train_y, val_X, val_y, test_X, test_y

    def _load_best_params(self) -> dict:
        """
        Load existing best parameters from the model_best_param file.

        :return: dict, existing best parameters
        """
        try:
            with open(self.paths["model_best_param"], "r") as file:
                existing_best_params = yaml.safe_load(file)
        except FileNotFoundError:
            existing_best_params = {"xgboost": {}, "log_reg": {}}
        return existing_best_params

    def _load_hyperparameters(self) -> dict:
        """
        Load hyperparameters from the configuration file.

        :return: dict, hyperparameters
        """
        with open(self.paths["model_hyperparameters"], "r") as file:
            return yaml.safe_load(file)

    def _save_best_params(self, best_params: dict) -> None:
        """
        Save the best parameters to the model_best_param file.

        :param best_params: dict, best parameters obtained from hyperparameter tuning
        """
        with open(self.paths["model_best_param"], "w") as file:
            yaml.dump(best_params, file)

    def hyperparameter_tuning_logreg(self, train_X, train_y):
        """
        Perform hyperparameter tuning for logistic regression using Grid Search Cross Validation.

        :param train_X: pd.DataFrame, training features
        :param train_y: pd.Series, training labels
        """
        existing_best_params = self._load_best_params()
        hyperparameters = self._load_hyperparameters()
        hyperparameters = hyperparameters["log_reg"]
        param_grid = {
            "penalty": hyperparameters["penalty"],
            "solver": hyperparameters["solver"],
            "max_iter": hyperparameters["max_iter"],
        }
        logreg_model = LogisticRegression()
        logreg_cv = GridSearchCV(
            logreg_model, param_grid, cv=3, verbose=0, scoring="accuracy"
        )
        logreg_cv.fit(train_X, train_y)
        existing_best_params["log_reg"] = logreg_cv.best_params_
        self._save_best_params(existing_best_params)

    def hyperparameter_tuning_xgboost(
        self, train_X: pd.DataFrame, train_y: pd.Series
    ) -> None:
        """
        Perform hyperparameter tuning for xgboost using Grid Search Cross Validation.

        :param train_X: pd.DataFrame, training features
        :param train_y: pd.Series, training labels
        """
        existing_best_params = self._load_best_params()
        hyperparameters = self._load_hyperparameters()
        hyperparameters = hyperparameters["xgboost"]
        param_grid = {
            "max_depth": hyperparameters["max_depth"],
            "learning_rate": hyperparameters["learning_rate"],
            "n_estimators": hyperparameters["n_estimators"],
        }
        xgb_model = xgb.XGBClassifier(eval_metric=["logloss"])
        xgb_cv = GridSearchCV(
            xgb_model, param_grid, cv=3, verbose=0, scoring="accuracy"
        )
        xgb_cv.fit(train_X, train_y)
        existing_best_params["xgboost"] = xgb_cv.best_params_
        self._save_best_params(existing_best_params)

    def _load_model_parameters(self) -> dict:
        """
        Load model parameters from the configuration file.

        :return: dict, model parameters
        """
        with open(self.paths["model_parameters"], "r") as file:
            return yaml.safe_load(file)

    def _save_model_object(self, model):
        """
        Save the trained model object to a file.

        :param model: trained model object
        """
        with open(self.paths["model_object"], "wb") as file:
            pickle.dump(model, file)

    def train_model_logreg(self, train_X: pd.DataFrame, train_y: pd.Series) -> None:
        """
        Train a logistic regression model and save it to a file.

        :param train_X: pd.DataFrame, training features
        :param train_y: pd.Series, training labels
        """
        model_parameters = self._load_model_parameters()["log_reg"]
        logreg_model = LogisticRegression(**model_parameters)
        logreg_model.fit(train_X, train_y)
        self._save_model_object(logreg_model)

    def train_model_xgboost(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        val_X: pd.DataFrame,
        val_y: pd.Series,
    ) -> None:
        """
        Train a XGBoost model and save it to a file.

        :param train_X: pd.DataFrame, training features
        :param train_y: pd.Series, training labels
        :param val_X: pd.DataFrame, validation features
        :param val_y: pd.Series, validation labels
        """
        model_parameters = self._load_model_parameters()["xgboost"]
        xgb_model = xgb.XGBClassifier(
            **model_parameters, eval_metric=["logloss", "error"]
        )
        xgb_model.fit(
            train_X, train_y, eval_set=[(train_X, train_y), (val_X, val_y)], verbose=0
        )
        self._save_model_object(xgb_model)

    def _load_model_object(self):
        """
        Load the trained model object from the file.

        :return: trained model object
        """
        with open(self.paths["model_object"], "rb") as file:
            return pickle.load(file)

    def model_predict(self, test_X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model object.

        :param test_X: pd.DataFrame, testing features for prediction
        :return: np.ndarray, predicted labels
        """
        model = self._load_model_object()
        return model.predict(test_X)

    def calculate_accuracy(
        self, test_y: pd.Series, predicted_values: np.ndarray
    ) -> float:
        """
        Calculate accuracy on the test set.

        :param test_y: pd.Series, true labels for the test set
        :param predicted_values: np.ndarray, predicted labels for the test set
        :return: float, accuracy on the test set
        """
        return accuracy_score(test_y, predicted_values)

    def plot_confusion_matrix(
        self, test_y: pd.Series, pred_y: np.ndarray, accuracy: float
    ) -> None:
        """
        Plot the confusion matrix with accuracy in the title.

        :param test_y: pd.Series, true labels for the test set
        :param pred_y: np.ndarray, predicted labels for the test set
        :param accuracy: float, accuracy on the test set
        """
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            confusion_matrix(test_y, pred_y),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%")
        plt.tight_layout()
        plt.savefig(self.paths["confusion_matrix"])
        plt.close()

    def save_predicted_output(
        self, test_X: pd.DataFrame, predicted_values: np.ndarray, test_y: pd.Series
    ) -> None:
        """
        Save the test set, predicted output, and the difference to an Excel file.

        :param test_X: pd.DataFrame, test features
        :param predicted_values: np.ndarray, predicted labels for the test set
        :param test_y: pd.Series, true labels for the test set
        """
        df_test = test_X.copy()
        df_test[self.column_names["predicted"]] = predicted_values
        df_test[self.column_names["target"]] = test_y
        df_test[self.column_names["difference"]] = (
            df_test[self.column_names["predicted"]]
            - df_test[self.column_names["target"]]
        )
        df_test.to_excel(self.paths["data_pred"], index=False)

    def plot_model_performance(self) -> None:
        """
        Plot the log loss and accuracy of the model during training.
        """
        model = self._load_model_object()
        results = model.evals_result()
        train_logloss = results["validation_0"]["logloss"]
        val_logloss = results["validation_1"]["logloss"]
        train_error = results["validation_0"]["error"]
        val_error = results["validation_1"]["error"]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_logloss, label="Train")
        plt.plot(val_logloss, label="Validation")
        plt.title("Log Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Log Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(1 - np.array(train_error), label="Train")
        plt.plot(1 - np.array(val_error), label="Validation")
        plt.title("Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.paths["model_performance"])
        plt.close()

    def plot_feature_importance(self) -> None:
        """
        Plot the feature importance of the trained XGBoost model.
        """
        model = self._load_model_object()
        xgb.plot_importance(model, importance_type="weight")
        plt.tight_layout()
        plt.savefig(self.paths["feature_importance"])
        plt.close()

    def plot_shapley_summary(self, test_X: pd.DataFrame) -> None:
        """
        Plot the Shapley summary plot for the XGBoost model.

        :param test_X: pd.DataFrame, test features
        """
        model = self._load_model_object()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_X)
        summary_plot = shap.summary_plot(
            shap_values,
            test_X,
            feature_names=test_X.columns,
            class_names=model.classes_,
            show=False,
        )
        plt.title("Shapley Summary Plot")
        plt.tight_layout()
        plt.savefig(self.paths["shapley_summary"])
        plt.close(summary_plot)

    def run_pipeline(self):
        """
        Run the end-to-end machine learning pipeline.
        """
        df = self.load_data()
        df = self.feature_engineering(df)
        self.save_data_description(df)
        self.plot_distribution_columns(df)
        self.plot_average_monthly_hours_vs_left(df)
        corr_matrix = self.calculate_pearson_correlation(df)
        self.plot_pearson_correlation_matrix(corr_matrix)
        df = self.drop_columns(df, self.column_names["to_drop"])
        df = self.one_hot_encode(df, self.column_names["to_encode"])
        train_data, val_data, test_data = self.split_data(df)
        train_X, train_y, val_X, val_y, test_X, test_y = self.prepare_data(
            train_data, val_data, test_data
        )
        if self.model_type == "log_reg":
            self.hyperparameter_tuning_logreg(train_X, train_y)
            self.train_model_logreg(train_X, train_y)
            pred_y = self.model_predict(test_X)
            accuracy = self.calculate_accuracy(test_y, pred_y)
            self.plot_confusion_matrix(test_y, pred_y, accuracy)
            self.save_predicted_output(test_X, pred_y, test_y)
        elif self.model_type == "xgboost":
            self.hyperparameter_tuning_xgboost(train_X, train_y)
            self.train_model_xgboost(train_X, train_y, val_X, val_y)
            pred_y = self.model_predict(test_X)
            accuracy = self.calculate_accuracy(test_y, pred_y)
            self.plot_confusion_matrix(test_y, pred_y, accuracy)
            self.save_predicted_output(test_X, pred_y, test_y)
            self.plot_model_performance()
            self.plot_feature_importance()
            self.plot_shapley_summary(test_X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predictive Analytics for Employee Attrition"
    )
    parser.add_argument(
        "-c",
        "--cfg_file",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        choices=["log_reg", "xgboost"],
        default="xgboost",
        help="Type of model to run: 'log_reg' or 'xgboost'",
    )
    args = parser.parse_args()

    EA = EmployeeAttrition(args.cfg_file, args.model_type)
    EA.run_pipeline()
