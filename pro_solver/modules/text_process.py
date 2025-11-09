import re
from typing import Dict, List, Tuple
from config.database.config import MATH_BLOCK_PATTERNS
import nbformat
import pathlib

def _mask_math(text: str) -> Tuple[str, Dict[str, str]]:
    masks = {}
    i = 0
    for pat, flags in MATH_BLOCK_PATTERNS:
        def repl(m):
            nonlocal i
            key = f"__MATHBLOCK_{i}__"
            masks[key] = m.group(0)
            i += 1
            return key
        text = re.sub(pat, repl, text, flags=flags)
    return text, masks

def _unmask_math(text: str, masks: Dict[str, str]) -> str:
    for k, v in masks.items():
        text = text.replace(k, v)
    return text

def chunk_latex(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    masked, masks = _mask_math(text)
    paras = [p for p in masked.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    cur_len = 0

    for p in paras:
        if len(p) > max_chars * 1.25:
            sentence_parts = re.split(r"(?<=[.!?])\s+", p)
            for sp in sentence_parts:
                if cur_len + len(sp) + 2 > max_chars and buf:
                    chunk = _unmask_math("\n\n".join(buf), masks)
                    chunks.append(chunk)

                    if overlap > 0:
                        tail = chunk[-overlap:]
                        buf = [tail]
                        cur_len = len(tail)
                    else:
                        buf, cur_len = [], 0
                buf.append(sp)
                cur_len += len(sp) + 2
        else:
            if cur_len + len(p) + 2 > max_chars and buf:
                chunk = _unmask_math("\n\n".join(buf), masks)
                chunks.append(chunk)
                if overlap > 0:
                    tail = chunk[-overlap:]
                    buf = [tail]
                    cur_len = len(tail)
                else:
                    buf, cur_len = [], 0
            buf.append(p)
            cur_len += len(p) + 2

    if buf:
        chunks.append(_unmask_math("\n\n".join(buf), masks))
    return chunks

def safe_read_text(path: pathlib.Path) -> str:
    try:
        if path.suffix == ".ipynb":
            with path.open("r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            parts = []
            for cell in nb.cells:
                if cell.cell_type == "markdown":
                    parts.append("# Markdown cell\n" + cell.source)
                elif cell.cell_type == "code":
                    parts.append("# Code cell\n" + cell.source)
            return f"# Notebook: {path.name}\n\n" + "\n\n".join(parts)
        else:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f""
    
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

import re, json

def safe_json_parse(text: str):
    cleaned = re.sub(r'```(?:json)?', '', text)
    cleaned = re.sub(r'^[^{]*', '', cleaned)
    cleaned = re.sub(r'[^}]*$', '', cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("⚠️ JSON parsing failed, cleaning quotes...")
        cleaned = cleaned.replace('\\"', '"').replace('"', '\\"').replace('\\"{', '{').replace('}\\"', '}')
        return json.loads(cleaned)