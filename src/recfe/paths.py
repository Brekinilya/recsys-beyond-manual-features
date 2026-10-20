from pathlib import Path


def _root() -> Path:
    # Prefer the working directory when it is the project root: on Windows the resolved
    # module path can be much longer (folder redirection) and hit the 260-char path limit.
    cwd = Path.cwd()
    pp = cwd / "pyproject.toml"
    if pp.exists() and 'name = "recfe"' in pp.read_text(encoding="utf-8"):
        return cwd
    return Path(__file__).resolve().parents[2]


ROOT = _root()
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "results"
FIGURES = REPORTS / "figures"


def processed(dataset: str) -> Path:
    p = PROCESSED / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def results(name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / name
