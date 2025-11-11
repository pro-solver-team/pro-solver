import re

MATH_BLOCK_PATTERNS = [
    (r"\$\$.*?\$\$", re.DOTALL),
    (r"\\\[.*?\\\]", re.DOTALL),
    (r"\\begin\{.*?\}.*?\\end\{.*?\}", re.DOTALL)
]

MAX_CHARS = 1200
OVERLAP = 150
CHUNK_SIZE = 1200
BATCH_SIZE = 512
