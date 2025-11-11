import re

DATASETS = [
    #"math-ai/StackMathQA",
    #"meta-math/MetaMathQA", 
    #"qwedsacf/competition_math",
    "TIGER-Lab/MathInstruct",
    #"open-r1/OpenR1-Math-220k"
]

FINITE_DIFF_REPOS = [
    "https://github.com/zwicker-group/py-pde",
]

NEURAL_REPOS = [
    "https://github.com/tum-pbs/PhiFlow",
    "https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks",
]

MODULUS_REPOS = [
    "https://github.com/NVIDIA/physicsnemo",
    "https://github.com/MehdiTaghizadehUVa/MFGNN_Flood",
]

ALL_REPOS = FINITE_DIFF_REPOS + NEURAL_REPOS + MODULUS_REPOS

INCLUDE = {".py", ".md", ".txt", ".ipynb"}

SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv", "env",
    "build", "dist", "site-packages", "node_modules", "data", "datasets",
    "images", "figures", "assets", "docs/_build"
}

MATH_BLOCK_PATTERNS = [
    (r"\$\$.*?\$\$", re.DOTALL),
    (r"\\\[.*?\\\]", re.DOTALL),
    (r"\\begin\{.*?\}.*?\\end\{.*?\}", re.DOTALL)
]

CHUNK_SIZE = 1200
OVERLAP = 200
BATCH_SIZE = 100

REPOS_LOAD_PATH = "../../../../data/chroma/repos"