"""Sentence-aware chunking for HotpotQA/2Wiki benchmark corpora."""
from __future__ import annotations
import json, argparse
from pathlib import Path

def make_chunks(documents, max_sentences=5, max_chars=1400):
    if max_sentences < 1 or max_chars < 1:
        raise ValueError("max_sentences and max_chars must be positive")
    chunks=[]
    for doc in documents:
        current=[]; current_chars=0
        def flush():
            if not current: return
            ids=[x["sent_id"] for x in current]
            chunks.append({
                "chunk_id": f"{doc['document_id']}:chunk:{ids[0]}-{ids[-1]}",
                "document_id": doc["document_id"], "dataset": doc["dataset"],
                "title": doc["title"], "text": " ".join(x["text"] for x in current),
                "sentence_ids": ids,
                "sentences": current.copy(),
                "question_ids": doc.get("question_ids", [])
            })
        for sentence in doc["sentences"]:
            text=sentence["text"]
            if current and (len(current)>=max_sentences or current_chars+len(text)>max_chars):
                flush(); current=[]; current_chars=0
            current.append({"sent_id": sentence["sent_id"], "text": text})
            current_chars += len(text)
        flush()
    return chunks

def main():
    p=argparse.ArgumentParser(description="Create sentence-aware benchmark chunks")
    p.add_argument("--input-dir", default="data/benchmark_corpus")
    p.add_argument("--output-dir", default="data/benchmark_chunks")
    p.add_argument("--max-sentences", type=int, default=5)
    p.add_argument("--max-chars", type=int, default=1400)
    a=p.parse_args(); out=Path(a.output_dir)
    for ds in ("hotpotqa","2wikimultihopqa"):
        docs=json.loads((Path(a.input_dir)/ds/"documents.json").read_text(encoding="utf-8"))
        chunks=make_chunks(docs,a.max_sentences,a.max_chars)
        target=out/ds/"chunks.json"; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"{ds}: {len(docs)} documents -> {len(chunks)} chunks -> {target}")
if __name__ == "__main__": main()
