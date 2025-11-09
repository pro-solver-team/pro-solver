from pro_solver.modules.dataset_process import to_q_a, make_doc_text, iter_rows
from pro_solver.modules.text_process import chunk_latex
from config.database.config import BATCH_SIZE
import uuid
from datasets import load_dataset, DatasetDict

def upsert_dataset(collection, hf_repo: str, limit: int | None):
    print(f"\n=== Loading {hf_repo} ===")

    try:
        data = load_dataset(hf_repo, split="train")
        splits = {"train": data}
    except Exception:
        dd: DatasetDict = load_dataset(hf_repo)
        splits = {k: v for k, v in dd.items()}

    total_added = 0
    for split_name, ds in splits.items():
        print(f" Split: {split_name} (size={len(ds)})")

        ids, docs, metas = [], [], []
        batch_counter = 0

        for idx, row in enumerate(iter_rows(ds, limit)):
            q, a, meta = to_q_a(hf_repo, row)
            if not q:
                continue
            text = make_doc_text(q, a)

            chunks = chunk_latex(text, max_chars=1200, overlap=150)
            base_id = str(row.get("id") or row.get("_id") or row.get("problem_id") or uuid.uuid4())

            for j, ch in enumerate(chunks):
                ids.append(f"{hf_repo}:{split_name}:{base_id}:{j}")
                docs.append(ch)
                m = dict(meta)
                m.update({
                    "split": split_name,
                    "chunk_index": j,
                    "num_chunks": len(chunks),
                })
                metas.append(m)

            if len(ids) >= BATCH_SIZE:
                collection.upsert(ids=ids, documents=docs, metadatas=metas)
                batch_counter += 1
                total_added += len(ids)
                ids, docs, metas = [], [], []

                if batch_counter % 5 == 0:
                    print(f"  ... upserted ~{total_added} chunks so far")

        if ids:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            total_added += len(ids)
            print(f"  ... final flush: total {total_added} chunks added for {hf_repo}")

    print(f" Done: {hf_repo}")