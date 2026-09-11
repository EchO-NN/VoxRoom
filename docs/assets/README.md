# VoxRoom media assets

The pipeline and simulation recording were supplied by the project maintainer for this release. Media are presented as qualitative illustrations; evaluation consumes saved label arrays, not these images.

| File | Content |
| --- | --- |
| `pipeline.pdf` | Original one-page pipeline figure, unchanged |
| `pipeline.png` | Raster rendering of that PDF for README display |
| `voxroom_simulation_10x.mp4` | Original supplied simulation recording, unchanged; 10× playback, 2200×1400, 30 fps, approximately 14.2 s |
| `simulation_preview.gif` | Reduced-size preview of the same video for GitHub README playback |
| `simulation_poster.jpg` | Single frame for the HTML video player's poster |

The simulation recording is illustrative and is not used to estimate module latency. The PDF's diagram labels the decision as `p(seed) > 0.5`; the actual implementation uses **p ≥ 0.5**, as described in the method and README. The supplied figure itself is not altered.

## Add the real-world recording later

1. Place the actual clip at `docs/assets/voxroom_real_robot.mp4` and a poster at `docs/assets/real_robot_poster.jpg`, or use a maintainer-approved external video URL if file size requires it.
2. Edit the `real-world` section of `docs/index.html`: replace the placeholder with a `<video controls playsinline preload="metadata">` element and the actual MP4 source/poster.
3. Replace the “Coming soon” paragraph in the top-level README with a preview and link. State whether playback is accelerated and identify the actual robot/scene.
4. Add the matching deployment code and instructions under `real_robot/`; do not relabel the existing ZED runtime as the new LiDAR/TensorRT implementation.

There is currently **no real-world video file or broken video link** in the page. No automatic request is made for a nonexistent recording.

## Rendering commands

```bash
pdftoppm -f 1 -singlefile -scale-to 2600 -png docs/assets/pipeline.pdf docs/assets/pipeline
ffmpeg -ss 9 -i docs/assets/voxroom_simulation_10x.mp4 -frames:v 1 -vf scale=1100:-1 docs/assets/simulation_poster.jpg
ffmpeg -i docs/assets/voxroom_simulation_10x.mp4 -filter_complex '[0:v]fps=8,scale=660:-1:flags=lanczos,split[v][p];[p]palettegen=max_colors=128[pal];[v][pal]paletteuse=dither=bayer' -loop 0 docs/assets/simulation_preview.gif
```

These commands render/convert supplied media; they do not synthesize experimental results. Git stores the current small media files directly; no Git LFS pointer is required.
