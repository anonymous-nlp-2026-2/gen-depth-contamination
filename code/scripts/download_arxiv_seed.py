"""Download arXiv abstracts via arXiv API. Smaller batches for reliability."""
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

OUTPUT = "/root/autodl-tmp/gen-depth-contamination/data/exp_024_arxiv/depth_0.jsonl"
TARGET = 5000
MIN_CHARS = 100
BATCH = 200
ARXIV_NS = "{http://www.w3.org/2005/Atom}"

docs = []
offset = 0
categories = ["cs.CL", "cs.LG", "cs.AI", "cs.CV", "stat.ML"]
query = "+OR+".join([f"cat:{c}" for c in categories])

print(f"Fetching arXiv abstracts, target={TARGET}")
sys.stdout.flush()

while len(docs) < TARGET:
    url = f"http://export.arxiv.org/api/query?search_query={query}&start={offset}&max_results={BATCH}&sortBy=submittedDate&sortOrder=descending"
    print(f"  offset={offset}, have={len(docs)}...", end="", flush=True)
    
    ok = False
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "exp024-seed/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                xml_data = resp.read().decode("utf-8")
            ok = True
            break
        except Exception as e:
            print(f" retry{attempt+1}({e})", end="", flush=True)
            time.sleep(5 * (attempt + 1))
    
    if not ok:
        print(" FAILED, stopping")
        break
    
    root = ET.fromstring(xml_data)
    entries = root.findall(f"{ARXIV_NS}entry")
    
    if not entries:
        print(f" no entries, stopping")
        break
    
    added = 0
    for entry in entries:
        summary_el = entry.find(f"{ARXIV_NS}summary")
        if summary_el is None or summary_el.text is None:
            continue
        abstract = summary_el.text.strip()
        abstract = " ".join(abstract.split())
        if len(abstract) < MIN_CHARS:
            continue
        docs.append({"text": abstract, "doc_id": len(docs)})
        added += 1
        if len(docs) >= TARGET:
            break
    
    print(f" got {added} ({len(docs)}/{TARGET})", flush=True)
    offset += BATCH
    time.sleep(3)

print(f"\nTotal: {len(docs)} abstracts")
if len(docs) == 0:
    print("FAILED!")
    sys.exit(1)

with open(OUTPUT, "w") as f:
    for doc in docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

lens = [len(d["text"]) for d in docs]
print(f"Written to {OUTPUT}")
print(f"Length: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.0f}")
print(f"Sample: {docs[0]['text'][:200]}")
print("DOWNLOAD_COMPLETE")
