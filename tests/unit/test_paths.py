from pathlib import Path

from scoring_model.runtime.paths import ProjectPaths

# ----------------------------------------------------------------------
# 1. Project root
# ----------------------------------------------------------------------

class TestProjectRoot:

    def test_default_root_is_repository_root(self):
        paths = ProjectPaths()
        assert (paths.root / "pyproject.toml").is_file()
        assert (paths.root / "src" / "scoring_model").is_dir()

    def test_custom_root_is_used(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        assert paths.root == tmp_path.resolve()

    def test_custom_root_accepts_string(self, tmp_path):
        paths = ProjectPaths(str(tmp_path))
        assert paths.root == tmp_path.resolve()
        assert isinstance(paths.root, Path)


# ----------------------------------------------------------------------
# 2. Sub directories
# ----------------------------------------------------------------------

class TestSubDirectories:

    def test_first_level_directories(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        root = tmp_path.resolve()
        assert paths.configs == root / "configs"
        assert paths.data == root / "data"
        assert paths.src == root / "src"
        assert paths.tests == root / "tests"

    def test_data_directories(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        assert paths.raw == paths.data / "raw"
        assert paths.processed == paths.data / "processed"
        assert paths.intermediate == paths.data / "intermediate"

    def test_package_directories(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        assert paths.scoring_model == paths.src / "scoring_model"
        assert paths.features == paths.scoring_model / "features"
        assert paths.models_registry == paths.scoring_model / "models" / "registry"
        assert paths.models_latest == paths.scoring_model / "models" / "latest"

    def test_reports_directories(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        assert paths.reports == paths.scoring_model / "reports"
        assert paths.figures == paths.reports / "figures"
        assert paths.metrics == paths.reports / "metrics"

    def test_no_directory_is_created(self, tmp_path):
        ProjectPaths(tmp_path)
        assert list(tmp_path.iterdir()) == []
