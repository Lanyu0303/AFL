import argparse
import os

from dataset.amazon_preprocess import _parse_gz


def find_meta_file(data_dir: str) -> str:
    for name in os.listdir(data_dir):
        if name.startswith("meta_") and name.endswith(".json.gz"):
            return os.path.join(data_dir, name)
    raise FileNotFoundError(f"No meta_*.json.gz found in {data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/amazon_beauty")
    args = parser.parse_args()

    meta_path = find_meta_file(args.data_dir)
    total = 0
    with_imurl = 0
    with_title = 0
    sample_urls = []

    for m in _parse_gz(meta_path):
        asin = m.get("asin")
        if not asin:
            continue
        total += 1
        title = (m.get("title") or "").strip()
        if title:
            with_title += 1
        url = (m.get("imUrl") or "").strip()
        if url:
            with_imurl += 1
            if len(sample_urls) < 5:
                sample_urls.append((asin, url))

    ratio = (with_imurl / total * 100.0) if total else 0.0
    print(f"meta file: {meta_path}")
    print(f"items total: {total}")
    print(f"items with title: {with_title}")
    print(f"items with imUrl: {with_imurl} ({ratio:.2f}%)")
    if sample_urls:
        print("sample imUrl entries:")
        for asin, url in sample_urls:
            print(f"  {asin} -> {url}")


if __name__ == "__main__":
    main()
