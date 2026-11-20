from math import log2

def gold_pairs(question):
    return {(str(x["title"]).strip().casefold(), int(x["sent_id"]))
            for x in question.get("supporting_facts", [])}

def retrieved_pairs(chunks):
    pairs=set()
    for c in chunks:
        title=str(c.get("title","")).strip().casefold()
        for sid in c.get("sentence_ids", []):
            pairs.add((title, int(sid)))
    return pairs

def score_retrieval(question, chunks, k_values=(5,10)):
    gold=gold_pairs(question)
    ranked=chunks
    if not gold:
        return {"retrieval_labeled": False}
    results={}
    for k in k_values:
        found=retrieved_pairs(ranked[:k]) & gold
        results[f"supporting_fact_recall@{k}"]=len(found)/len(gold)
        results[f"supporting_fact_precision@{k}"]=len(found)/max(1,len(retrieved_pairs(ranked[:k])))
    rr=0.0
    for rank,c in enumerate(ranked,1):
        if retrieved_pairs([c]) & gold:
            rr=1.0/rank
            break
    results["mrr"]=rr
    # Chunk-level binary relevance for ranking quality. Sentence-level coverage
    # is measured separately by supporting_fact_recall@K.
    gains=[1.0 if (retrieved_pairs([c]) & gold) else 0.0 for c in ranked[:10]]
    dcg=sum(g/log2(i+2) for i,g in enumerate(gains))
    relevant_titles={title for title, _ in gold}
    ideal=[1.0]*min(len(relevant_titles),10)
    idcg=sum(g/log2(i+2) for i,g in enumerate(ideal))
    results["ndcg@10"]=dcg/idcg if idcg else 0.0
    return results
