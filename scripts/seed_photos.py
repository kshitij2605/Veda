#!/usr/bin/env python3
"""Seed a real-image 'photos' collection for the multimodal showcase.

Fetches real photos, computes the CLIP *image* embedding (512-d) + a zero-shot
category (classifying each image against text labels via CLIP — so text->image
search is demonstrable), and inserts {image_vec, image(base64), caption, category}.
The base64 lives in a plain string field so the UI can render it as a data: URI.
"""
import base64
import json
import urllib.error
import urllib.request

CLIP = "http://localhost:8100/v1/embeddings"
VEDA = "http://localhost:6340"
LABELS = [
    "a landscape or mountains",
    "a city or buildings",
    "a person or portrait",
    "food or drink",
    "an animal",
    "a vehicle",
    "plants and nature",
    "an abstract texture",
]


def clip(body):
    r = urllib.request.Request(CLIP, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))


def veda(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(VEDA + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def fetch(seed):
    url = f"https://picsum.photos/seed/{seed}/256/256"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return resp.read()


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


lab = clip({"model": "clip", "input": LABELS})
lvecs = [d["embedding"] for d in lab["data"]]

name = "photos"
veda("DELETE", f"/v1/collections/{name}")
fields = [
    {"name": "image_vec", "type": "vector", "dimension": 512, "metric": "cosine", "model": "clip-vit-base-patch32"},
    {"name": "image", "type": "string"},
    {"name": "caption", "type": "string"},
    {"name": "category", "type": "string", "filterable": True},
]
print("create:", veda("POST", "/v1/collections", {"name": name, "fields": fields, "shard_count": 1}))

seeds = [
    "alps", "tokyo-night", "golden-gate", "ocean-wave", "pine-forest", "sahara", "fish-market",
    "portrait7", "old-bridge", "zen-garden", "sunset22", "wood-cabin", "rainy-street", "blue-lake",
    "wildflowers", "vintage-car", "espresso", "snow-peak", "river-bend", "neon-city",
]
recs = []
for s in seeds:
    try:
        img = fetch(s)
    except Exception as e:
        print("skip", s, e)
        continue
    b64 = base64.b64encode(img).decode()
    ie = clip({"model": "clip", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": "Represent the given image."}]}]})
    iv = ie["data"][0]["embedding"]
    bi = max(range(len(LABELS)), key=lambda i: dot(iv, lvecs[i]))
    cat = LABELS[bi]
    recs.append({"image_vec": iv, "image": b64, "caption": f"{cat.capitalize()} · {s}", "category": cat})
    print("embedded", s, "->", cat)

st, body = veda("POST", f"/v1/collections/{name}/insert", {"records": recs})
print("insert:", st, body)
