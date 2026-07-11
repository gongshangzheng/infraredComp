"""论文路由 — 直接操作本地数据库，独立于 SeekVerse。"""
import io
import os
import urllib.request
from typing import Optional

from fastapi import APIRouter, Body, Response

from server.config import THUMBNAILS_DIR
from server.db import (
    init_db, query_papers, get_paper_by_id, get_stats,
    set_starred, set_pinned, get_note, save_note, set_blog_url,
)

router = APIRouter(prefix="/api/papers", tags=["papers"])

# 启动时初始化数据库
init_db()


def _render_arxiv_thumbnail_webp(arxiv_id: str, pdf_url: str | None = None) -> bytes | None:
    """Fetch an arXiv PDF, render its first page to WebP, cache and return bytes.

    Returns None on any failure (network / parse / render). Cached file is
    reused on subsequent requests for the same arxiv_id.
    """
    os.makedirs(THUMBNAILS_DIR, exist_ok=True)
    cache_path = os.path.join(THUMBNAILS_DIR, f"{arxiv_id}.webp")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return f.read()
        except OSError:
            pass

    url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        import pymupdf
        from PIL import Image

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        pdf_data = urllib.request.urlopen(req, timeout=30).read()
        doc = pymupdf.open(stream=pdf_data, filetype="pdf")
        if doc.page_count == 0:
            return None
        pix = doc[0].get_pixmap(dpi=110)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=80)
        webp = buf.getvalue()
        with open(cache_path, "wb") as f:
            f.write(webp)
        return webp
    except Exception:  # noqa: BLE001 - thumbnail is best-effort
        return None


@router.get("")
async def list_papers(limit: int = 100, offset: int = 0, source: Optional[str] = None, category: Optional[str] = None):
    """获取论文列表。"""
    papers, total = query_papers(source=source, category=category, limit=limit, offset=offset)
    return {"total": total, "papers": papers}


@router.get("/stats/summary")
async def get_stats_summary():
    """获取论文统计。"""
    return get_stats()


@router.get("/{paper_id}")
async def get_paper_detail(paper_id: str):
    """获取论文详情。"""
    paper = get_paper_by_id(paper_id)
    return paper if paper else {"error": "not found"}


@router.get("/{paper_id}/note")
async def get_paper_note(paper_id: str):
    """获取论文笔记。"""
    return get_note(paper_id)


@router.put("/{paper_id}/note")
async def save_paper_note(paper_id: str, content: str = Body(..., embed=True)):
    """保存论文笔记。"""
    return save_note(paper_id, content)


@router.put("/{paper_id}/blog")
async def set_blog_url_route(paper_id: str, blog_url: str = Body(..., embed=True)):
    """设置论文 blog 链接。"""
    return set_blog_url(paper_id, blog_url)


@router.put("/{paper_id}/star")
async def toggle_star(paper_id: str, starred: bool = Body(..., embed=True)):
    """收藏 / 取消收藏。"""
    set_starred(paper_id, starred)
    return {"status": "ok", "starred": starred}


@router.put("/{paper_id}/pin")
async def toggle_pin(paper_id: str, pinned: bool = Body(..., embed=True)):
    """置顶 / 取消置顶。"""
    set_pinned(paper_id, pinned)
    return {"status": "ok", "pinned": pinned}


@router.post("/{paper_id}/summarize")
async def summarize_paper(paper_id: str):
    """触发论文摘要生成（暂未实现，返回占位响应）。"""
    return {"status": "not_implemented", "message": "AI 摘要功能尚未接入"}


# ========== 缩略图 ==========

@router.get("/thumbnails/{arxiv_id}")
async def get_thumbnail(arxiv_id: str):
    """获取论文首页缩略图(WebP)。

    懒加载:首次请求时从 arXiv 抓 PDF,用 PyMuPDF 渲染首页为 WebP 并缓存到
    papers/cache/thumbnails/{arxiv_id}.webp;后续直接返回缓存。失败返回 404。
    """
    webp = _render_arxiv_thumbnail_webp(arxiv_id)
    if not webp:
        return Response(content=b"", status_code=404)
    return Response(content=webp, media_type="image/webp")
