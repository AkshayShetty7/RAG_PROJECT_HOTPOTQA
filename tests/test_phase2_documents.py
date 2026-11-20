import json, tempfile, unittest
from pathlib import Path
from data_processing.prepare_benchmark_documents import prepare_documents
from rag.benchmark_chunker import make_chunks

class Phase2Tests(unittest.TestCase):
    def test_deduplication_and_sentence_ids_are_preserved(self):
        payload={"questions":[
            {"id":"h1","dataset":"hotpotqa","contexts":[{"title":"Page A","sentences":["one","two"]}]},
            {"id":"h2","dataset":"hotpotqa","contexts":[{"title":"Page A","sentences":["one","two"]}]},
            {"id":"w1","dataset":"2wikimultihopqa","contexts":[{"title":"Page A","sentences":["other"]}]}
        ]}
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"b.json"; src.write_text(json.dumps(payload),encoding="utf-8")
            summary=prepare_documents(src,Path(td)/"out")
            h=json.loads((Path(td)/"out/hotpotqa/documents.json").read_text())
            w=json.loads((Path(td)/"out/2wikimultihopqa/documents.json").read_text())
            self.assertEqual(summary["hotpotqa"]["documents"],1)
            self.assertEqual(len(h),1); self.assertEqual(len(w),1)
            self.assertEqual([x["sent_id"] for x in h[0]["sentences"]],[0,1])
            self.assertEqual(len(h[0]["question_ids"]),2)

    def test_chunks_keep_original_sentence_ids(self):
        docs=[{"document_id":"d1","dataset":"hotpotqa","title":"A","sentences":[
            {"sent_id":0,"text":"first"},{"sent_id":1,"text":"second"},{"sent_id":2,"text":"third"}],"question_ids":["q1"]}]
        chunks=make_chunks(docs,max_sentences=2,max_chars=100)
        self.assertEqual([c["sentence_ids"] for c in chunks],[[0,1],[2]])
        self.assertEqual(chunks[0]["text"],"first second")

if __name__ == "__main__": unittest.main()
