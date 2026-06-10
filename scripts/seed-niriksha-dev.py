#!/usr/bin/env python3
"""Seed Niriksha dev data: multimodal-shaped collections + synthetic vectors."""
import json, random, urllib.request, urllib.error
BASE = "http://localhost:6340"
random.seed(7)

def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

# (collection, [(field, dim, model)], [(scalar, type, filterable)], n_records)
COLLECTIONS = [
    ("articles",        [("text_embedding", 1024, "BAAI/bge-m3")],
        [("title", "string", False), ("category", "string", True)], 42),
    ("product_images",  [("image_embedding", 768, "Qwen3-VL-Embedding-2B")],
        [("caption", "string", False), ("sku", "string", True)], 36),
    ("voice_notes",     [("audio_embedding", 512, "laion/clap")],
        [("transcript", "string", False)], 21),
    ("video_clips",     [("video_embedding", 1024, "videoclip")],
        [("title", "string", False)], 15),
    ("contracts",       [("doc_embedding", 1024, "vidore/colpali")],
        [("filename", "string", True)], 27),
    ("product_catalog", [("text_embedding", 1024, "BAAI/bge-m3"), ("image_embedding", 768, "Qwen3-VL-Embedding-2B")],
        [("name", "string", False), ("price", "float32", True), ("in_stock", "bool", True)], 50),
]

def main():
    for name, vecs, scalars, n in COLLECTIONS:
        req("DELETE", f"/v1/collections/{name}")
        fields = [{"name": f, "type": "vector", "dimension": d, "metric": "cosine", "model": m} for f, d, m in vecs]
        for f, t, filt in scalars:
            fields.append({"name": f, "type": t, "filterable": filt})
        st, body = req("POST", "/v1/collections", {"name": name, "fields": fields, "shard_count": 1})
        print("create", name, st, body)
        recs = []
        for i in range(n):
            rec = {}
            for f, d, _ in vecs:
                rec[f] = [round(random.uniform(-1, 1), 4) for _ in range(d)]
            for f, t, _ in scalars:
                rec[f] = (f"{name}-{f}-{i}" if t == "string" else
                          round(random.uniform(1, 999), 2) if t.startswith("float") else
                          (i % 2 == 0) if t == "bool" else i)
            recs.append(rec)
        st, body = req("POST", f"/v1/collections/{name}/insert", {"records": recs})
        print("insert", name, st, body)

if __name__ == "__main__":
    main()
