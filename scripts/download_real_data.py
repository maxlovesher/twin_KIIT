"""Download the open ShanghaiT2DM dataset used for real-patient validation.

Source: Zhao Q, Zhu J, Shen X, et al. "Chinese diabetes datasets for data-driven machine learning."
Scientific Data 10, 35 (2023). Figshare DOI 10.6084/m9.figshare.20444397. License: CC BY 4.0.

    python scripts/download_real_data.py      # -> data/external/shanghai/  (~3.4 MB)
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glucotwin import config as C  # noqa: E402

ARTICLE = "https://api.figshare.com/v2/articles/20444397"
DEST = C.DATA_DIR / "external" / "shanghai"


def main() -> None:
    meta = json.load(urllib.request.urlopen(ARTICLE))
    f = meta["files"][0]
    print(f"downloading {f['name']} ({f['size'] / 1e6:.1f} MB, {meta['license']['name']}) ...")
    blob = urllib.request.urlopen(f["download_url"]).read()
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or len(parts) < 2 or Path(info.filename).name.startswith("~$"):
                continue
            target = DEST.joinpath(*parts[1:])  # drop the archive's top-level folder (non-UTF-8 name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    n = len(list((DEST / "Shanghai_T2DM").glob("*.xls*")))
    print(f"done: {n} T2DM recordings in {DEST}")


if __name__ == "__main__":
    main()
