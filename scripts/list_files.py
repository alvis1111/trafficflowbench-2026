import json, os, urllib.request, sys, time

TOKEN = os.environ.get("KGAT_TOKEN", "")
BASE = "https://www.kaggle.com/api/v1/competitions/data/list/2026-ieee-big-data-traffic-flow-bench?pageSize=200"

allf = []
page = BASE
for i in range(60):
    req = urllib.request.Request(page, headers={"Authorization": "Bearer " + TOKEN})
    d = json.load(urllib.request.urlopen(req, timeout=90))
    files = d["files"]
    allf.extend(files)
    nt = d.get("nextPageTokenNullable")
    print(f"page {i}: +{len(files)} files (total {len(allf)}), next={bool(nt)}", flush=True)
    if not nt:
        break
    page = BASE + "&pageToken=" + nt
    time.sleep(0.2)

tot = sum(f["totalBytes"] for f in allf)
print(f"\nTOTAL FILES: {len(allf)}")
print(f"TOTAL SIZE: {tot/1e6:.1f} MB")
json.dump(allf, open("meta/filelist_full.json", "w"))
print("saved meta/filelist_full.json")
