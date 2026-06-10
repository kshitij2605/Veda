import base64
import io
import json
import urllib.request

from PIL import Image


def post(body):
    r = urllib.request.Request(
        "http://localhost:8100/v1/embeddings",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(r, timeout=60))


t = post({"model": "clip", "input": "a photo of a cat"})
print("text dim:", len(t["data"][0]["embedding"]))

buf = io.BytesIO()
Image.new("RGB", (32, 32), (200, 30, 30)).save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()
im = post(
    {
        "model": "clip",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                    {"type": "text", "text": "Represent the given image."},
                ],
            }
        ],
    }
)
print("image dim:", len(im["data"][0]["embedding"]))
print("OK both paths work")
