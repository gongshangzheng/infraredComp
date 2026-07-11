"""Two-stage contour-video compression benchmark.

Stage 1: extract a lossless grayscale contour frame sequence from a raw video
         (pluggable edge extractor, e.g. Canny / Sobel).
Stage 2: compress the contour video with standard video codecs (x264 / x265 /
         svtav1 / vp9) at several CRF levels and evaluate quality + speed.

The two stages are decoupled by the lossless contour PNG sequence, which serves
as both the stage-2 input and the quality ground truth.
"""
