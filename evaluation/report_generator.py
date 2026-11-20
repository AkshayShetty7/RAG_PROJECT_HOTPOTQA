import json
from pathlib import Path
import pandas as pd

def main():
    p=Path("results/raw/retrieval_metrics.jsonl")
    rows=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    records=[]
    for method in ("dense","bm25","hybrid","hybrid_rerank"):
        for ds in ("hotpotqa","2wikimultihopqa"):
            subset=[r for r in rows if r["dataset"]==ds]
            metrics=[r["retrieval"][method]["metrics"] for r in subset]
            keys=sorted({k for m in metrics for k in m if k!="retrieval_labeled"})
            rec={"dataset":ds,"method":method,"questions":len(subset)}
            for key in keys:
                vals=[m[key] for m in metrics if key in m]
                rec[key]=sum(vals)/len(vals) if vals else None
            records.append(rec)
    out=Path("results/reports"); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/"retrieval_summary.csv",index=False)
    print(out/"retrieval_summary.csv")

if __name__=="__main__": main()
