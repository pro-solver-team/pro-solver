INCLUDE = {".py", ".md", ".txt", ".ipynb"}

SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv", "env",
    "build", "dist", "site-packages", "node_modules", "data", "datasets",
    "images", "figures", "assets", "docs/_build"
}

CHUNK_SIZE = 1200
OVERLAP = 200
BATCH_SIZE = 100

REPOS_LOAD_PATH = "../../../../data/chroma/repos"
