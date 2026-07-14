"""快速 benchmark：在真实 imagenet 数据路径上扫 batch size，找 GPU 不闲 + 显存安全的 bs。

用 shards=1 的一个小子集（max_images=1024）跑固定步数，测 samples/s + peak VRAM，
覆盖 JPEG 解码 + canny + ELIC forward/backward 的真实流水线。
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from scripts.imagenet_contour_dataset import ImageNetContourDataset
from scripts.train_model import build_model, rd_loss


def bench(bs, nw=4, steps=25, warmup=5):
    dev = "cuda"
    ds = ImageNetContourDataset(split="train", method="canny", max_images=1024, size=128, shards=1, rg_cache_cap=(128 if nw <= 0 else max(8, 160 // max(1, nw))))
    kw = {} if nw <= 0 else {"num_workers": nw, "persistent_workers": True, "prefetch_factor": 4}
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, **kw)
    model = build_model("ELIC", 1, dev, warm_start=True).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    it = iter(loader)
    # warmup
    for _ in range(warmup):
        x = next(it).to(dev)
        opt.zero_grad(); out = model.forward(x); loss, *_ = rd_loss(out, x, 0.01); loss.backward(); opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); n = 0
    for _ in range(steps):
        try:
            x = next(it).to(dev)
        except StopIteration:
            it = iter(loader); x = next(it).to(dev)
        opt.zero_grad(); out = model.forward(x); loss, *_ = rd_loss(out, x, 0.01); loss.backward(); opt.step(); n += bs
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    sps = n / dt
    del model, opt, loader, ds
    torch.cuda.empty_cache()
    return sps, peak


if __name__ == "__main__":
    print(f"{'bs':>4} {'samples/s':>10} {'peak_vram_GB':>13}")
    for bs in [16, 32, 64, 96, 128, 160]:
        try:
            sps, peak = bench(bs)
            print(f"{bs:>4} {sps:>10.1f} {peak:>13.1f}")
        except Exception as e:
            print(f"{bs:>4} ERROR: {e}")
