from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time

TITLE_RE = re.compile(rb'"wikipedia_title": "((?:[^"\\]|\\.)*)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=os.environ.get("PEARL_ASEARCHER_PAGES_PATH"),
                    help="Path to ASearcher-Local-Knowledge's wiki_webpages.jsonl. "
                         "Defaults to PEARL_ASEARCHER_PAGES_PATH.")
    ap.add_argument("--out_dir", default="cache/asearcher_page_index")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    titles: dict[str, tuple[int, int]] = {}
    n_dup = 0
    t0 = time.time()
    pos = 0
    with open(args.pages, "rb") as f:
        for i, line in enumerate(f):
            m = TITLE_RE.search(line, 0, 2048)
            title = json.loads(b'"' + m.group(1) + b'"') if m else json.loads(line)["wikipedia_title"]
            if title in titles:
                n_dup += 1
            else:
                titles[title] = (pos, len(line))
            pos += len(line)
            if i % 1_000_000 == 0:
                print(f"{i:,} pages, {pos / 1e9:.1f} GB, {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(args.out_dir, "titles.pkl"), "wb") as f:
        pickle.dump(titles, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done: {len(titles):,} titles ({n_dup:,} duplicate titles skipped), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
