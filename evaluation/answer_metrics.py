import re
import string
from collections import Counter

def normalize_answer(s):
    s=str(s or "").lower()
    s="".join(ch for ch in s if ch not in string.punctuation)
    s=re.sub(r"\b(a|an|the)\b"," ",s)
    return " ".join(s.split())

def exact_match(prediction, reference):
    return float(normalize_answer(prediction)==normalize_answer(reference))

def token_f1(prediction, reference):
    p=normalize_answer(prediction).split()
    r=normalize_answer(reference).split()
    if not p or not r:
        return float(p==r)
    common=Counter(p)&Counter(r)
    overlap=sum(common.values())
    if not overlap: return 0.0
    precision=overlap/len(p); recall=overlap/len(r)
    return 2*precision*recall/(precision+recall)
