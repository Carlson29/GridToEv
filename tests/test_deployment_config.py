import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTests(unittest.TestCase):
    def test_daily_model_is_opt_in_for_image_and_compose_but_enabled_on_render(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        render = (ROOT / "render.yaml").read_text(encoding="utf-8")

        self.assertNotIn("GRIDTOEV_DAILY_MODEL_PATH=/app/models/v2/", dockerfile)
        self.assertIn(
            'GRIDTOEV_DAILY_MODEL_PATH: "${GRIDTOEV_DAILY_MODEL_PATH:-}"',
            compose,
        )
        self.assertIn(
            'key: GRIDTOEV_DAILY_MODEL_PATH\n        value: "/app/models/v2/daily_curtailment_bundle.joblib"',
            render,
        )
        self.assertIn(
            "COPY --chown=gridtoev:gridtoev models/v2/daily_curtailment_bundle.joblib",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
