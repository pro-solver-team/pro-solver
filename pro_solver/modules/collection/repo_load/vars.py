import re
from pathlib import Path

DB_DIR = Path(__file__).resolve().parents[3]/'data'

COLLECTION_NAME = "math"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512

DATASETS = [
    "math-ai/StackMathQA",
    "meta-math/MetaMathQA",
    "qwedsacf/competition_math",
    "TIGER-Lab/MathInstruct",
    "open-r1/OpenR1-Math-220k"
]
MAX_RECORDS_PER_DATASET = 3

FINITE_DIFF_REPOS = [
    "https://github.com/zwicker-group/py-pde",
]

NEURAL_GITHUB_REPOS = [
    "https://github.com/tum-pbs/PhiFlow",
    "https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks",
]

MODULUS_GITHUB_REPOS = [
    "https://github.com/NVIDIA/physicsnemo",
    "https://github.com/MehdiTaghizadehUVa/MFGNN_Flood",
]

INCLUDE = {".py", ".md", ".txt", ".ipynb"}

SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv", "env",
    "build", "dist", "site-packages", "node_modules", "data", "datasets",
    "images", "figures", "assets", "docs/_build"
}

ALL_REPOS = FINITE_DIFF_REPOS + NEURAL_GITHUB_REPOS + MODULUS_GITHUB_REPOS

MATH_BLOCK_PATTERNS = [
    (r"\$\$.*?\$\$", re.DOTALL),
    (r"\\\[.*?\\\]", re.DOTALL),
    (r"\\begin\{.*?\}.*?\\end\{.*?\}", re.DOTALL)
]

PDF_PATHS = [
            'pdfs/Nonlin_methods.pdf',
            'pdfs/numeric.pdf'
            ]

CHUNK_SIZE = 1200
OVERLAP = 200
BATCH_SIZE = 100

REPOS_LOAD_PATH = DB_DIR/"data"/"chroma"/"repos"