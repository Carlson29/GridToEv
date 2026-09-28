"""The published fitted rules must reproduce the loaded estimators, not just name them."""

import math
import unittest

import numpy as np
from fastapi.testclient import TestClient

from gridtoev.api import create_app
from gridtoev.constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from gridtoev.daily_model import DailyCurtailmentService
from gridtoev.fitted_formulas import export_v1_tree, export_v2_tree
from gridtoev.inference import PredictionService
from gridtoev.release_guard import DEFAULT_REPORT_PATH


def _tree_value(tree: dict, row: np.ndarray) -> float:
    nodes = {node["node_id"]: node for node in tree["nodes"]}
    for item in nodes.values():
        numeric = item["value"] if item["is_leaf"] else item["threshold"]
        if not math.isfinite(numeric):
            raise AssertionError("A fitted tree exposed a non-finite leaf or threshold")
    node = nodes[0]
    while not node["is_leaf"]:
        value = row[node["feature_index"]]
        go_left = node["missing_go_to_left"] if np.isnan(value) else value <= node["threshold"]
        node = nodes[node["left_child"] if go_left else node["right_child"]]
    return node["value"]


class FittedFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = PredictionService(DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH, DEFAULT_REPORT_PATH)
        cls.v2 = DailyCurtailmentService()
        cls.client = TestClient(create_app(cls.v1, api_key="test-key", daily_service=cls.v2))
        cls.client.__enter__()
        cls.headers = {"X-API-Key": "test-key"}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_manifest_identifies_every_fitted_estimator_and_input(self) -> None:
        for model, path, expected_count in (
            ("v1", "/model-info/v1/fitted-formulas", 7),
            ("v2", "/model-info/daily-curtailment/fitted-formulas", 2),
        ):
            self.assertEqual(self.client.get(path).status_code, 401)
            response = self.client.get(path, headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["model_id"], model)
            self.assertEqual(len(payload["estimators"]), expected_count)
            self.assertEqual(len(payload["input_features"]), 119 if model == "v1" else 24)
            self.assertEqual(len(payload["transformed_features"]), len(payload["input_features"]))
            self.assertTrue(all(item["tree_count"] > 0 for item in payload["estimators"]))
            self.assertTrue(all(item["training_target"] and item["prediction_output"] for item in payload["estimators"]))
            self.assertTrue(all("{tree_index}" in item["tree_endpoint_template"] for item in payload["estimators"]))
            self.assertNotIn("curtailment_mwh", [item["name"] for item in payload["input_features"]])

        v1 = self.client.get("/model-info/v1/fitted-formulas", headers=self.headers).json()
        estimators = {item["id"]: item for item in v1["estimators"]}
        self.assertEqual(estimators["curtailment_regressor"]["inverse_link"], "exp")
        self.assertEqual(estimators["constraint_regressor"]["inverse_link"], "exp")
        self.assertEqual(estimators["dispatch_down_regressor"]["serving_weight_by_horizon"], {"30": 0.0, "60": 0.0})
        self.assertIn("median", v1["preprocessing"].lower())
        self.assertEqual(v1["serving_formulas_endpoint"], "/model-info/v1/formulas")

        v2 = self.client.get("/model-info/daily-curtailment/fitted-formulas", headers=self.headers).json()
        amount = {item["id"]: item for item in v2["estimators"]}["amount_model"]
        self.assertEqual(amount["tree_count"], 240)
        self.assertEqual(amount["aggregation"], "mean")
        self.assertIn("positive", amount["training_target"].lower())
        for model, path in (("v1", "/model-info/v1/fitted-formulas"), ("v2", "/model-info/daily-curtailment/fitted-formulas")):
            about = {item["model_id"]: item for item in self.client.get("/models/about", headers=self.headers).json()["models"]}
            catalog = {item["model_id"]: item for item in self.client.get("/models/catalog", headers=self.headers).json()["models"]}
            serving = "/model-info/v1/formulas" if model == "v1" else "/model-info/daily-curtailment/formulas"
            self.assertEqual(about[model]["fitted_formulas_endpoint"], path)
            self.assertEqual(catalog[model]["fitted_formulas_endpoint"], path)
            self.assertEqual(self.client.get(serving, headers=self.headers).json()["fitted_formulas_endpoint"], path)

    def test_tree_routes_expose_numeric_splits_and_bound_indices(self) -> None:
        cases = (
            ("/model-info/v1/fitted-formulas/event_classifier/trees/0", "v1"),
            ("/model-info/daily-curtailment/fitted-formulas/amount_model/trees/0", "v2"),
        )
        for path, model in cases:
            self.assertEqual(self.client.get(path).status_code, 401)
            response = self.client.get(path, headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["model_id"], model)
            self.assertEqual(payload["tree_index"], 0)
            self.assertTrue(payload["nodes"])
            self.assertEqual(payload["nodes"][0]["node_id"], 0)
            for node in payload["nodes"]:
                if not node["is_leaf"]:
                    self.assertIsNotNone(node["feature_name"])
                    self.assertIsNotNone(node["threshold"])
                    self.assertIsNotNone(node["left_child"])
                    self.assertIsNotNone(node["right_child"])
        for path in (
            "/model-info/v1/fitted-formulas/not-a-model/trees/0",
            "/model-info/v1/fitted-formulas/event_classifier/trees/99999",
            "/model-info/daily-curtailment/fitted-formulas/amount_model/trees/-1",
        ):
            self.assertEqual(self.client.get(path, headers=self.headers).status_code, 404)

    def test_exported_tree_rules_reconstruct_loaded_estimators(self) -> None:
        v1_columns = self.v1.bundle["feature_columns"]
        v1_row = self.v1.dataset.iloc[[100]][v1_columns]
        for name in self.v1.bundle["models"]:
            pipeline = self.v1.bundle["models"][name]
            transformed = pipeline.named_steps["imputer"].transform(v1_row)[0]
            model = pipeline.named_steps["model"]
            score = float(model._baseline_prediction[0, 0]) + sum(
                _tree_value(export_v1_tree(self.v1.bundle, name, index), transformed)
                for index in range(model.n_iter_)
            )
            expected = float(pipeline.predict_proba(v1_row)[0, 1]) if name == "event_classifier" else float(pipeline.predict(v1_row)[0])
            actual = 1 / (1 + math.exp(-score)) if name == "event_classifier" else (math.exp(score) if name in {"curtailment_regressor", "constraint_regressor"} else score)
            self.assertAlmostEqual(actual, expected, places=8, msg=name)

        v2_columns = self.v2.bundle["metadata"]["feature_columns"]
        v2_row = self.v2.dataset.iloc[[100]][v2_columns]
        v2_values = v2_row.to_numpy(dtype=float)[0]
        classifier = self.v2.bundle["event_model"]
        score = float(classifier._baseline_prediction[0, 0]) + sum(
            _tree_value(export_v2_tree(self.v2.bundle, "event_model", index), v2_values)
            for index in range(classifier.n_iter_)
        )
        self.assertAlmostEqual(1 / (1 + math.exp(-score)), float(classifier.predict_proba(v2_row)[0, 1]), places=8)
        amount = self.v2.bundle["amount_model"]
        mean = sum(
            _tree_value(export_v2_tree(self.v2.bundle, "amount_model", index), v2_values)
            for index in range(len(amount.estimators_))
        ) / len(amount.estimators_)
        self.assertAlmostEqual(mean, float(amount.predict(v2_row)[0]), places=8)


if __name__ == "__main__":
    unittest.main()
