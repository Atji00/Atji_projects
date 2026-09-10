from pathlib import Path


class ProjectPaths:

    def __init__(self, path: str | Path | None = None) -> None:

        """Class to manage project paths."""

        self.root = Path(__file__).resolve().parents[3] if path is None else Path(path).resolve()

        self.configs = self.root / "configs"
        self.data = self.root / "data"
        self.src = self.root / "src"
        self.tests = self.root / "tests"

        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        self.intermediate = self.data / "intermediate"

        self.scoring_model = self.src / "scoring_model"

        self.dataset = self.scoring_model / "dataset"
        self.features = self.scoring_model / "features" 
        self.models = self.scoring_model / "models"
        self.runtime = self.scoring_model / "runtime"
        self.scripts = self.scoring_model / "scripts"

        self.models_latest = self.models / "latest"
        self.models_registry = self.models / "registry"

        self.reports = self.scoring_model / "reports"
        self.figures = self.reports / "figures"
        self.metrics = self.reports / "metrics"

