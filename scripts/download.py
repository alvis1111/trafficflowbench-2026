import json, os, sys, time, urllib.request, urllib.parse, concurrent.futures

TOKEN = os.environ.get("KGAT_TOKEN", "")
BASE = "https://www.kaggle.com/api/v1/competitions/data/download/2026-ieee-big-data-traffic-flow-bench"

# Which panels to download (can extend later). None = skip that split.
PANELS = ["D12_I405_S"]
SPLITS = ["train"]  # Task 1 is only self-scoreable on train

def wanted(name):
    p = name
    if p.startswith("kaggle_public/config/"):
        return True
    for panel in PANELS:
        pre = f"kaggle_public/corridors/{panel}/"
        if p.startswith(pre + "network/"):
            return True
        for sp in SPLITS:
            if p.startswith(pre + sp + "/mainline_states/"):
                return True
            if p.startswith(pre + sp + "/mainline_states_masked/"):
                return True
        # Task1 template (train)
        if p.startswith(f"kaggle_public/task1/{panel}/train/"):
            return True
    return False

allf = json.load(open("meta/filelist_full.json"))
selected = [f for f in allf if wanted(f["name"])]
tot = sum(f["totalBytes"] for f in selected)
print(f"selected {len(selected)} files, {tot/1e6:.1f} MB", flush=True)

def dl(f):
    name = f["name"]
    enc = urllib.parse.quote(name, safe="")
    url = BASE + "/" + enc
    local = os.path.join("release", name.replace("/", os.sep))
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if os.path.exists(local) and os.path.getsize(local) == f["totalBytes"]:
        return (name, 0, "skip")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + TOKEN})
            data = urllib.request.urlopen(req, timeout=120).read()
            with open(local, "wb") as fh:
                fh.write(data)
            return (name, len(data), "ok")
        except Exception as e:
            if attempt == 3:
                return (name, 0, "fail")
            time.sleep(1.5 * (attempt + 1))
    return (name, 0, "fail")

done = 0
fail = 0
failed_names = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    for name, n, status in ex.map(dl, selected):
        done += 1
        if status == "fail":
            fail += 1
            failed_names.append(name)
        if done % 25 == 0 or done == len(selected):
            print(f"  {done}/{len(selected)}  (fail={fail})", flush=True)
print(f"DONE: {done} files, {fail} failed", flush=True)
for n in failed_names:
    print("  FAILED:", n, flush=True)
