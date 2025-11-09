import os
import pathlib
from typing import List
from git import Repo, GitCommandError
import nbformat
from pro_solver.modules.text_process import chunk_text, safe_read_text
from config.database.config import INCLUDE_EXTS, SKIP_DIRS

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
        return ""

def iter_repo_files(root: pathlib.Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            if p.suffix.lower() in INCLUDE_EXTS:
                yield p

def shallow_clone(url: str, dest_root: pathlib.Path) -> pathlib.Path:
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = dest_root / repo_name
    dest_root.mkdir(parents=True, exist_ok=True)
    
    if dest.exists() and (dest / ".git").exists():
        try:
            Repo(dest).git.pull("--ff-only")
        except GitCommandError:
            pass
        return dest

    Repo.clone_from(url, dest, depth=1, no_single_branch=True)
    return dest

def add_repos_to_chroma(collection, repo_urls: List[str], batch_size: int = 100):
    repos_root = pathlib.Path("/content/repos")
    total_chunks = 0

    for url in repo_urls:
        print(f"Processing {url} ...")
        repo_path = shallow_clone(url, repos_root)
        repo_rel_base = repo_path.name

        to_add_docs, to_add_ids, to_add_metas = [], [], []

        for fpath in iter_repo_files(repo_path):
            rel_path = str(fpath.relative_to(repo_path))
            raw = safe_read_text(fpath)
            if not raw or raw.strip() == "":
                continue

            header = f"# File: {rel_path}\n"
            text = header + raw

            chunks = chunk_text(text, chunk_size=1200, overlap=200)
            for i, ch in enumerate(chunks):
                doc_id = f"{repo_rel_base}:{rel_path}:{i}"
                to_add_docs.append(ch)
                to_add_ids.append(doc_id)
                to_add_metas.append({
                    "repo": url,
                    "repo_name": repo_rel_base,
                    "path": rel_path,
                    "chunk_index": i,
                })

                if len(to_add_ids) >= batch_size:
                    try:
                        collection.upsert(
                            documents=to_add_docs, 
                            metadatas=to_add_metas, 
                            ids=to_add_ids
                        )
                    except Exception as e:
                        print(f"Batch error: {e}")
                    total_chunks += len(to_add_ids)
                    to_add_docs, to_add_ids, to_add_metas = [], [], []

        if to_add_ids:
            try:
                collection.upsert(
                    documents=to_add_docs, 
                    metadatas=to_add_metas, 
                    ids=to_add_ids
                )
            except Exception as e:
                print(f"Final batch error for {url}: {e}")
            total_chunks += len(to_add_ids)

        print(f"{url} -> added ~{total_chunks} chunks so far.")