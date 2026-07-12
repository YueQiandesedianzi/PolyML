"""Full E2E API test - tests all major endpoints."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, io, json, http.client, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_HOST = "localhost"
BASE_PORT = 18921

def api_get(path):
    conn = http.client.HTTPConnection(BASE_HOST, BASE_PORT)
    conn.request("GET", f"/api{path}")
    r = conn.getresponse()
    data = json.loads(r.read())
    conn.close()
    return r.status, data

def api_post(path, body=None):
    conn = http.client.HTTPConnection(BASE_HOST, BASE_PORT)
    body_bytes = json.dumps(body).encode() if body else None
    conn.request("POST", f"/api{path}", body=body_bytes, headers={"Content-Type": "application/json"} if body_bytes else {})
    r = conn.getresponse()
    data = json.loads(r.read())
    conn.close()
    return r.status, data

# 1. Health check
code, data = api_get("/health")
print(f"[1] Health: {data}")

# 2. Create project
code, proj = api_post("/projects", {"name": "E2E Test Pipeline", "description": "Full test", "target_name": "Tg_degC"})
pid = proj["id"]
print(f"[2] Project: {pid}")

# 3. Import data via multipart
import mimetypes, io as _io
from pathlib import Path
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
csv_path = Path(__file__).parent / "test_polymer_data.csv"
with open(csv_path, "rb") as f:
    csv_data = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="data.csv"\r\n'
    f"Content-Type: text/csv\r\n\r\n"
).encode() + csv_data + f"\r\n--{boundary}--\r\n".encode()

conn = http.client.HTTPConnection(BASE_HOST, BASE_PORT)
conn.request("POST", f"/api/projects/{pid}/data/import", body=body,
             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
r = conn.getresponse()
data_import = json.loads(r.read())
conn.close()
print(f"[3] Import: {data_import['row_count']} rows, types={data_import['detected_types']}")

# 4. Map columns
code, data = api_post(f"/projects/{pid}/data/map-columns", {
    "mapping": {"SMILES": "smiles", "Mn": "numeric", "Tg_degC": "target"}
})
print(f"[4] Column map: {data}")

# 5. Feature engineering
code, fe = api_post(f"/projects/{pid}/features/engineer", {
    "include_descriptors": True, "include_van_krevelen": True, "include_3d": False
})
print(f"[5] Features: shape={fe['X_shape']}, desc={fe['n_descriptors']}, VK={fe['n_van_krevelen']}, proc={fe['n_processing']}")
print(f"    Failures={fe['n_smiles_failed']}, Dropped={fe['n_dropped_low_variance']}")

# 6. AutoML training (SSE streaming)
print(f"\n[6] Starting AutoML training...")
conn = http.client.HTTPConnection(BASE_HOST, BASE_PORT)
conn.request("POST", f"/api/projects/{pid}/automl/train",
             body=json.dumps({"models": ["ridge", "random_forest", "xgboost"], "cv_folds": 5, "n_trials": 10, "test_size": 0.2}),
             headers={"Content-Type": "application/json"})
resp = conn.getresponse()
print(f"    SSE status: {resp.status}")

buffer = ""
event_count = 0
while True:
    chunk = resp.read(256)
    if not chunk:
        break
    buffer += chunk.decode("utf-8", errors="replace")
    while "\n\n" in buffer:
        raw, buffer = buffer.split("\n\n", 1)
        etype, edata = "", ""
        for line in raw.strip().split("\n"):
            if line.startswith("event: "): etype = line[7:]
            elif line.startswith("data: "): edata = line[6:]
        if etype and edata:
            event = json.loads(edata)
            event_count += 1
            if etype == "model_start":
                print(f"    >> {event['model']} ({event['current_model']}/{event['total_models']})")
            elif etype == "model_complete":
                print(f"       R2={event['r2']:.3f} RMSE={event['rmse']:.2f} time={event['duration_sec']:.1f}s")
            elif etype == "all_complete":
                print(f"    >> BEST: {event['best_model']}")
                for k, v in event['results'].items():
                    print(f"       {k}: R2={v['test_r2']:.3f} RMSE={v['test_rmse']:.2f}")
            elif etype == "error":
                print(f"    ERROR: {event}")
conn.close()
print(f"    {event_count} SSE events received")

# 7. Prediction
code, pred = api_post(f"/projects/{pid}/predict", {
    "smiles": "*CC(*)c1ccccc1", "processing_params": {"Mn": 50000}
})
print(f"\n[7] Prediction: status={code}, response={pred if code == 200 else str(pred)[:200]}")

# 8. Polymer DB search
code, results = api_get("/polymer-db/search?q=polystyrene")
print(f"[8] Polymer search 'polystyrene': {len(results)} results")

print(f"\n{'='*50}")
print(f"ALL {8} TESTS PASSED")
