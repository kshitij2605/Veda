#!/usr/bin/env python3
"""Seed a real text 'docs' collection for Search v2: real BGE-M3 dense
embeddings (via embed-insert against the live endpoint) + bm25-indexed
title/body so Dense, BM25, and Hybrid search are all meaningful."""
import json
import urllib.error
import urllib.request

VEDA = "http://localhost:6340"


def veda(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(VEDA + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


DOCS = [
    ("Vector databases explained", "A vector database stores high-dimensional embeddings and finds nearest neighbors for semantic search, recommendations, and retrieval-augmented generation.", "technology"),
    ("Sourdough bread basics", "Sourdough relies on a wild yeast starter; long fermentation develops a tangy flavor and an open, chewy crumb.", "food"),
    ("The James Webb telescope", "JWST observes the universe in infrared, peering through cosmic dust to image the earliest galaxies and exoplanet atmospheres.", "space"),
    ("Jazz improvisation", "Improvisation lets jazz musicians spontaneously compose melodies over chord changes, trading solos and reacting to the rhythm section.", "music"),
    ("Marathon training", "Building endurance for a marathon means gradual weekly mileage increases, long runs, tempo work, and adequate recovery.", "sports"),
    ("Compound interest", "Compound interest grows savings exponentially as returns are reinvested, rewarding early and consistent investing.", "finance"),
    ("Mediterranean diet", "Rich in olive oil, vegetables, fish, and whole grains, the Mediterranean diet is linked to heart health and longevity.", "health"),
    ("Northern lights", "The aurora borealis glows when solar particles excite gases in the upper atmosphere, painting the polar sky green and violet.", "travel"),
    ("Transformer neural networks", "Transformers use self-attention to weigh relationships between tokens, powering modern language and vision models.", "technology"),
    ("Roman aqueducts", "Roman engineers moved water across vast distances using gravity-fed aqueducts with precise, gentle gradients.", "history"),
    ("Espresso extraction", "A good espresso shot balances pressure, grind size, and timing to pull a sweet, syrupy crema in about 25 seconds.", "food"),
    ("Black holes", "A black hole's gravity is so strong that not even light escapes beyond its event horizon.", "space"),
    ("Electric vehicles", "EVs replace combustion engines with battery-electric drivetrains, cutting tailpipe emissions and maintenance.", "technology"),
    ("Yoga and flexibility", "Regular yoga improves flexibility, balance, and breath control while reducing stress.", "health"),
    ("Coral reefs", "Coral reefs host a quarter of marine life but are threatened by warming seas and ocean acidification.", "nature"),
    ("Index funds", "Low-cost index funds track a market benchmark, offering broad diversification with minimal fees.", "finance"),
    ("Classical guitar", "Nylon-string classical guitar is played fingerstyle, prized for its warm tone and expressive dynamics.", "music"),
    ("Hiking the Alps", "Alpine trails wind past glaciers, meadows, and mountain huts, rewarding hikers with panoramic summit views.", "travel"),
    ("Quantum computing", "Quantum computers use qubits and superposition to explore many states at once, promising speedups for certain problems.", "technology"),
    ("Fermented foods", "Kimchi, yogurt, and miso are fermented foods that introduce beneficial bacteria and complex umami flavors.", "food"),
    ("Tennis serve", "A powerful tennis serve combines a high toss, leg drive, and pronation to generate pace and spin.", "sports"),
    ("Renaissance art", "Renaissance painters revived perspective, anatomy, and light, producing lifelike works like the Mona Lisa.", "history"),
    ("Cloud object storage", "Object storage in the cloud scales to billions of files with durability, versioning, and pay-as-you-go pricing.", "technology"),
    ("Migratory birds", "Migratory birds navigate thousands of miles using the sun, stars, and Earth's magnetic field.", "nature"),
]

name = "docs"
veda("DELETE", f"/v1/collections/{name}")
fields = [
    {"name": "embedding", "type": "vector", "dimension": 1024, "metric": "cosine", "model": "BAAI/bge-m3"},
    {"name": "title", "type": "string", "bm25": True},
    {"name": "body", "type": "string", "bm25": True},
    {"name": "category", "type": "string", "filterable": True},
]
print("create:", veda("POST", "/v1/collections", {"name": name, "fields": fields, "shard_count": 1}))

recs = [{"title": t, "body": b, "category": c} for (t, b, c) in DOCS]
st, body = veda("POST", f"/v1/collections/{name}/embed-insert", {"records": recs, "auto_embed": {"embedding": {"source_field": "body"}}})
print("embed-insert:", st, body)
