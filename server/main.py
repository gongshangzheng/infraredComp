"""FastAPI 主入口 — infraredComp 后端(镜像 ProjFlow server/main.py)。"""
import sys
import os

# 确保能 import server 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.config import CORS_ORIGINS
from server.routers import management, papers, benchmark

app = FastAPI(
    title="infraredComp API",
    description="红外/轮廓视频压缩评测平台后端:项目管理、论文搜集、轮廓视频压缩评测",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(management.router)
app.include_router(papers.router)
app.include_router(benchmark.router)


@app.get("/")
async def root():
    return {"message": "infraredComp API", "docs": "/docs"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8090, reload=True)
