#!/usr/bin/env python3
"""把 infraredComp 现有静态论文数据导入本地 SQLite 数据库。

数据源: web/src/data/papers.json(原静态前端使用的 24 篇种子论文)
目标:   data/papers.db(ProjFlow 式 schema,见 server/db.py)

字段映射(保持 schema 不增列,数据不丢):
  id/title/title_zh/abstract/abstract_zh/authors/url/pdf_url/source/categories -> 直接映射
  arxiv_id            -> external_ids.arxiv
  github_url/venue/blog_slug/published_at_str/tags -> metadata.*
  其余 ProjFlow 字段(relevance_score 等)取默认值
"""
import json
import sys
from datetime import datetime
from pathlib import Path

# 确保能 import server 模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from server.db import init_db, upsert_paper

PAPERS_JSON = Path(__file__).parent.parent / "web" / "src" / "data" / "papers.json"


def map_paper(p: dict) -> dict:
    """把 papers.json 的一条记录映射到 db.py 的 paper_data dict。"""
    arxiv_id = p.get("arxiv_id", "")
    paper_id = p.get("id") or (f"arxiv-{arxiv_id}" if arxiv_id else None)
    if not paper_id:
        # 兜底:用标题 hash
        paper_id = f"manual-{abs(hash(p.get('title', ''))) % 10**12}"

    authors = p.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    if not authors:
        authors = ["Unknown"]

    metadata = {
        "github_url": p.get("github_url", ""),
        "venue": p.get("venue", ""),
        "blog_slug": p.get("blog_slug", ""),
        "published_at_str": p.get("published_at_str", ""),
        "tags": p.get("tags", []),
    }

    external_ids = {}
    if arxiv_id:
        external_ids["arxiv"] = arxiv_id

    return {
        "id": paper_id,
        "title": p.get("title", "") or "Untitled",
        "title_zh": p.get("title_zh", ""),
        "abstract": p.get("abstract", ""),
        "abstract_zh": p.get("abstract_zh", ""),
        "authors": authors,
        "published_at": p.get("published_at", ""),
        "crawled_at": datetime.now().isoformat(),
        "url": p.get("url", ""),
        "pdf_url": p.get("pdf_url", ""),
        "source": p.get("source", "manual") or "manual",
        "external_ids": external_ids,
        "metadata": metadata,
        "categories": p.get("categories", []),
        "relevance_score": 0.5,
        "llm_classification": [],
        "arxiv_categories": [],
        "starred": False,
        "pinned": False,
    }


def main() -> None:
    if not PAPERS_JSON.exists():
        print(f"ERROR: {PAPERS_JSON} not found")
        sys.exit(1)

    print(f"Reading {PAPERS_JSON}...")
    with open(PAPERS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    papers = data.get("papers", []) if isinstance(data, dict) else data
    print(f"  Found {len(papers)} papers")

    init_db()
    print(f"Importing into SQLite via upsert_paper...")
    inserted = 0
    for p in papers:
        try:
            upsert_paper(map_paper(p))
            inserted += 1
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {p.get('id', '?')}: {e}")
    print(f"  Imported: {inserted}")

    # 统计
    import sqlite3
    from server.db import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    cats = conn.execute(
        "SELECT category, COUNT(*) FROM paper_categories GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()

    print("\n=== Import Complete ===")
    print(f"Total papers in database: {total}")
    print("Category distribution:")
    for cat, count in cats:
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
