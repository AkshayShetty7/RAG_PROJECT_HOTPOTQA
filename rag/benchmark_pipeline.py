"""Benchmark-specific retrieval pipeline; preserves chunk metadata."""
import json
from pathlib import Path
from dataclasses import dataclass
from rag.embedder import Embedder
from rag.bm25_search import BM25Search
from sentence_transformers import CrossEncoder
import faiss
import numpy as np

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    dataset: str
    title: str
    text: str
    sentence_ids: list
    sentences: list
    question_ids: list

    @classmethod
    def from_dict(cls, d):
        return cls(**{k:d.get(k, [] if k in ("sentence_ids","sentences","question_ids") else "") 
                      for k in ("chunk_id","document_id","dataset","title","text","sentence_ids","sentences","question_ids")})

class BenchmarkRetriever:
    def __init__(self, dataset, index_root="data/benchmark_indexes",
                 embedder=None, reranker=None):
        if dataset not in ("hotpotqa","2wikimultihopqa"):
            raise ValueError("Unknown dataset namespace")
        self.dataset=dataset
        self.path=Path(index_root)/dataset
        raw=json.loads((self.path/"chunks.json").read_text(encoding="utf-8"))
        self.chunks=[Chunk.from_dict(x) for x in raw]
        self.by_id={c.chunk_id:c for c in self.chunks}
        self.index=faiss.read_index(str(self.path/"faiss.index"))
        self.bm25 = BM25Search(self.chunks)
        self.embedder=embedder or Embedder()
        self.reranker=reranker

    def dense(self, query, k=10):
        q=np.asarray(self.embedder.encode_query(query),dtype=np.float32)
        scores, ids=self.index.search(q,min(k,self.index.ntotal))
        out=[]
        for s,i in zip(scores[0],ids[0]):
            if i>=0:
                out.append({"chunk":self.chunks[int(i)],"score":float(s)})
        return out

    def sparse(self, query, k=10):
        results=self.bm25.search(query,top_k=k)
        out=[]
        for x in results:
            raw=x["chunk"]
            cid=getattr(raw,"chunk_id",None)
            if cid in self.by_id:
                out.append({"chunk":self.by_id[cid],"score":float(x["score"])})
        return out

    @staticmethod
    def fuse(dense, sparse, k=10, rrf_k=60):
        scores={}; lookup={}
        for result_list in (dense,sparse):
            for rank,item in enumerate(result_list,1):
                c=item["chunk"]; cid=c.chunk_id
                scores[cid]=scores.get(cid,0.0)+1.0/(rrf_k+rank)
                lookup[cid]=c
        ids=sorted(scores,key=scores.get,reverse=True)[:k]
        return [{"chunk":lookup[cid],"score":scores[cid]} for cid in ids]

    def retrieve(self, query, method="hybrid", top_k=10, rerank=False):
        if method=="dense": results=self.dense(query,top_k)
        elif method=="bm25": results=self.sparse(query,top_k)
        elif method=="hybrid":
            results=self.fuse(self.dense(query,20),self.sparse(query,20),top_k)
        else: raise ValueError(f"Unknown method: {method}")
        if rerank:
            if self.reranker is None:
                self.reranker=CrossEncoder("BAAI/bge-reranker-base")
            pairs=[(query,x["chunk"].text) for x in results]
            scores=self.reranker.predict(pairs)
            results=[{"chunk":x["chunk"],"score":float(s)} for x,s in zip(results,scores)]
            results.sort(key=lambda x:x["score"],reverse=True)
            results=results[:top_k]
        return results
