"""Build separate FAISS and BM25 indexes for each benchmark corpus.

Requires the project's existing sentence-transformers, faiss-cpu and rank-bm25
installations. Run from the project root: python -m rag.build_benchmark_indexes
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.bm25_search import BM25Search

MODEL_NAME = "BAAI/bge-base-en-v1.5"

def build_indexes(chunks_dir, index_dir, model_name=MODEL_NAME, batch_size=64):
    embedder=Embedder(model_name)
    base=Path(chunks_dir); out=Path(index_dir)
    summary={}
    for ds in ("hotpotqa","2wikimultihopqa"):
        chunks=json.loads((base/ds/"chunks.json").read_text(encoding="utf-8"))
        if not chunks: raise ValueError(f"No chunks found for {ds}")
        texts=[c["text"] for c in chunks]
        vectors=[]
        for start in range(0,len(texts),batch_size):
            vectors.append(embedder.encode_documents(texts[start:start+batch_size]))
        embeddings=np.concatenate(vectors,axis=0).astype("float32")
        store=VectorStore(embeddings.shape[1]); store.add(embeddings,chunks)
        target=out/ds; target.mkdir(parents=True,exist_ok=True)
        store.save(target/"faiss.index")
        (target/"chunks.json").write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding="utf-8")
        bm25=BM25Search([_ChunkView(c) for c in chunks])
        with (target/"bm25.pkl").open("wb") as f: pickle.dump(bm25,f)
        summary[ds]={"chunks":len(chunks),"embedding_dimension":int(embeddings.shape[1]),"index_dir":str(target)}
        print(f"{ds}: indexed {len(chunks)} chunks at {target}")
    return summary

class _ChunkView:
    """Adapter exposing .text and .chunk_id expected by BM25Search/HybridSearch."""
    def __init__(self, record):
        self.__dict__.update(record)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--chunks-dir",default="data/benchmark_chunks")
    p.add_argument("--index-dir",default="data/benchmark_indexes")
    p.add_argument("--model",default=MODEL_NAME)
    p.add_argument("--batch-size",type=int,default=64)
    a=p.parse_args(); build_indexes(a.chunks_dir,a.index_dir,a.model,a.batch_size)
if __name__ == "__main__": main()
