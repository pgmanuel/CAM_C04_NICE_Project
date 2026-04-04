from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path
    snomed_path: Path
    chroma_persist_dir: Path
    audit_dir: Path
    regression_cases_path: Path


def find_project_root(start_path: Path | None = None) -> Path:
    current = (start_path or Path(__file__)).resolve().parent
    while current != current.parent:
        if (current / "snomed_master_v3.csv").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not resolve project root from snomed_master_v3.csv")


def build_project_paths(base_dir: Path) -> ProjectPaths:
    return ProjectPaths(
        base_dir=base_dir,
        snomed_path=base_dir / "snomed_master_v3.csv",
        chroma_persist_dir=base_dir / "chroma_db",
        audit_dir=base_dir / "audit",
        regression_cases_path=base_dir / "CAM_Project" / "tests" / "regression_cases.json",
    )
