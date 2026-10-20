"""Download raw files for both datasets.

Yambda: only the 50M flat multi-event log plus artist/album mappings
(audio embeddings, 13.8 GB, are not used). KION: original (Russian) archive.
"""
import shutil
import urllib.request
import zipfile

from recfe.paths import RAW

YAMBDA = "https://huggingface.co/datasets/yandex/yambda/resolve/main/"
YAMBDA_FILES = ["flat/50m/multi_event.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet"]
KION_ZIP = "https://raw.githubusercontent.com/irsafilo/KION_DATASET/main/data_original.zip"


def fetch(url, dst):
    if dst.exists():
        print("exists:", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dst)


def main():
    for f in YAMBDA_FILES:
        fetch(YAMBDA + f, RAW / "yambda" / f.split("/")[-1])
    z = RAW / "kion" / "data_original.zip"
    fetch(KION_ZIP, z)
    out = RAW / "kion" / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            # take only the three csv files, flattened, ignoring __MACOSX junk
            if name.startswith("data_original/") and name.endswith(".csv"):
                target = out / name.split("/")[-1]
                if not target.exists():
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    print("done")


if __name__ == "__main__":
    main()
