import os
import requests

out = "data/raw/olta_tr"
os.makedirs(out, exist_ok=True)
base = "https://zenodo.org/records/19472163/files/{}?download=1"

for name in ["phishing.csv", "legitimate.csv"]:
    r = requests.get(base.format(name), timeout=300)
    r.raise_for_status()
    with open(f"{out}/{name}", "wb") as f:
        f.write(r.content)
    print(name, len(r.content) // 1024, "KB")