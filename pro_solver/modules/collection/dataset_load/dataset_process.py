from typing import Dict, Any, Iterable, List, Tuple
from datasets import Dataset

def pick_first(d: Dict[str, Any], candidates: List[str]) -> str:
    for c in candidates:
        if c in d and d[c] and isinstance(d[c], str) and d[c].strip():
            return d[c]
    return ""

def to_q_a(dataset_name: str, row: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    dn = dataset_name.lower()

    if ("tiger-lab" in dn) and ("mathinstruct" in dn):
        q = pick_first(row, ["instruction", "question", "problem", "prompt", "input", "query", "title"])
        a = pick_first(row, ["output", "solution", "answer", "response", "rationale", "cot", "explanation"])
        meta = {
            "dataset": dataset_name,
            "source": str(row.get("source", "")),
            "subset": str(row.get("subset", row.get("type", ""))),
            "id_in_source": str(row.get("id") or row.get("_id") or row.get("problem_id") or row.get("qid") or ""),
        }
        return q, a, meta

    if ("open-r1" in dn) and ("openr1-math-220k" in dn):
        q = pick_first(row, ["problem", "question", "prompt"])
        a = pick_first(row, ["solution"])
        
        if not a:
            gens = row.get("generations")
            if isinstance(gens, list) and gens:
                a = "\n\n---\n\n".join([g for g in gens if isinstance(g, str) and g.strip()])
            if not a and isinstance(row.get("messages"), list):
                msgs = [m.get("content", "") for m in row["messages"]
                        if isinstance(m, dict) and m.get("content")]
                if msgs:
                    a = "\n\n".join(msgs)

        meta = {
            "dataset": dataset_name,
            "answer": str(row.get("answer", "")),
            "problem_type": str(row.get("problem_type", "")),
            "question_type": str(row.get("question_type", "")),
            "source": str(row.get("source", "")),
            "uuid": str(row.get("uuid", "")),
        }
        return q, a, meta

    q = pick_first(row, ["question", "problem", "instruction", "prompt", "title", "query"])
    a = pick_first(row, ["solution", "answer", "output", "response", "explanation"])
    meta = {"dataset": dataset_name}
    return q, a, meta

def make_doc_text(q: str, a: str) -> str:
    q = q.strip()
    a = a.strip()

    return (
        "\\section*{Problem}\n"
        + q
        + ("\n\n\\section*{Solution}\n" + a if a else "")
    )

def iter_rows(ds: Dataset, limit: int | None) -> Iterable[Dict[str, Any]]:
    n = len(ds)
    count = n if limit is None else min(limit, n)
    for i in range(count):
        yield ds[i]
