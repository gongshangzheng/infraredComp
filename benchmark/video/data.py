"""Data models for the contour-video benchmark.

`ContourArtifact` is the lossless grayscale PNG frame sequence produced by
stage 1 (the contour video) plus its manifest. `VideoCompressionResult` is one
row of stage-2 output (one codec @ one CRF on one sequence) and is the unit
persisted to results.json and consumed by the FastAPI backend.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ContourArtifact:
    """A lossless contour frame sequence (the stage-1 product / stage-2 input)."""

    source_name: str               # stem of the raw video / frame dir
    method: str                    # extractor name (canny / sobel / ...)
    frames_dir: str                # dir holding frame_%06d.png
    frame_paths: list[str] = field(default_factory=list)  # ordered PNG paths
    frame_count: int = 0
    fps: float = 25.0
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    manifest_path: str = ""        # manifest.json path

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoCompressionResult:
    """One codec @ one CRF on one contour sequence."""

    id: str                        # f"{sequence_name}|{codec}|crf{crf}"
    codec: str                     # e.g. "x264"
    codec_family: str              # e.g. "h264"
    crf: int
    sequence_name: str
    method: str                    # extractor that produced the contour video
    frame_count: int
    fps: float
    width: int
    height: int
    psnr: float
    ssim: float
    per_frame_psnr: list[float] = field(default_factory=list)
    per_frame_ssim: list[float] = field(default_factory=list)
    bitrate_kbps: float = 0.0
    bpp: float = 0.0
    compression_ratio: float = 0.0
    compressed_bytes: int = 0
    duration_s: float = 0.0
    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    enc_fps: float = 0.0
    dec_fps: float = 0.0
    temporal_metric: float = 0.0  # std of per-frame PSNR (lower = more stable)
    decoded_sample: str = ""       # path to one reconstructed frame, for the demo
    dataset: str = ""              # dataset this run belongs to (e.g. "Xiph-CIF-natural")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
