import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


def load_id2imgurl(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "::" not in line:
                continue
            iid, url = line.split("::", 1)
            mapping[int(iid)] = url.strip()
    return mapping


def download_image(item_id, url, img_dir, timeout=10):
    if not url:
        return item_id, False, "no_url"
    ext = url.split(".")[-1].split("?")[0].lower()
    if ext not in ("jpg", "jpeg", "png", "gif", "webp"):
        ext = "jpg"
    out_path = os.path.join(img_dir, f"{item_id}.{ext}")
    if os.path.exists(out_path):
        return item_id, True, "cached"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return item_id, True, "ok"
        return item_id, False, f"http_{resp.status_code}"
    except Exception as e:
        return item_id, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id2imgurl", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    id2url = load_id2imgurl(args.id2imgurl)
    print(f"total ids in id2imgurl: {len(id2url)}")

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_image, iid, url, args.img_dir, args.timeout) for iid, url in id2url.items()]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            _, ok, _ = fut.result()
            if ok:
                success += 1
            else:
                failed += 1

    print(f"downloaded/cached: {success}")
    print(f"failed/no_url: {failed}")


if __name__ == "__main__":
    main()
