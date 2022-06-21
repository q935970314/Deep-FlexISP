import numpy as np


class Evaluator:

    def __init__(self):
        monitored_metrics = ["mean", "median", "trimean", "bst25", "wst25", "wst5"]
        self.__metrics = {}
        self.__best_metrics = {m: 100.0 for m in monitored_metrics}
        self.__errors = []

    def add_error(self, error: float):
        self.__errors.append(error)

    def reset_errors(self):
        self.__errors = []

    def get_errors(self) -> list:
        return self.__errors

    def get_metrics(self) -> dict:
        return self.__metrics

    def get_best_metrics(self) -> dict:
        return self.__best_metrics

    def compute_metrics(self) -> dict:
        self.__errors = sorted(self.__errors)
        self.__metrics = {
            "mean": np.mean(self.__errors),
            "median": self.__g(0.5),
            "trimean": 0.25 * (self.__g(0.25) + 2 * self.__g(0.5) + self.__g(0.75)),
            "bst25": np.mean(self.__errors[:int(0.25 * len(self.__errors))]),
            "wst25": np.mean(self.__errors[int(0.75 * len(self.__errors)):]),
            "wst5": self.__g(0.95)
        }
        return self.__metrics

    def update_best_metrics(self) -> dict:
        self.__best_metrics["mean"] = self.__metrics["mean"]
        self.__best_metrics["median"] = self.__metrics["median"]
        self.__best_metrics["trimean"] = self.__metrics["trimean"]
        self.__best_metrics["bst25"] = self.__metrics["bst25"]
        self.__best_metrics["wst25"] = self.__metrics["wst25"]
        self.__best_metrics["wst5"] = self.__metrics["wst5"]
        return self.__best_metrics

    def __g(self, f: float) -> float:
        return np.percentile(self.__errors, f * 100)
