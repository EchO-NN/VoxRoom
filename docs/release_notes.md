# Paper-oriented repository update — September 2026

The project is **submitted to ICRA 2027 and under review**. This status does not imply acceptance.

## Maintained release

The release brings together the current VoxRoom source, the English project README, method/reproduction documentation, the original supplied pipeline PDF and its PNG rendering, the supplied 10× simulation MP4, a GIF preview, and a self-contained browser demo page. `real_robot/` reserves the separate LiDAR/TensorRT deployment code; the real-world video remains pending.

Current defaults remain line correlation 0.95, orthogonal variance 0.65 cells², and NN threshold 0.5. Experimental L-shaped dual-arm processing is off. The tested per-column encoding-reuse optimization is retained. No model retraining is performed by this documentation/release update.

## Paper tables versus archived experiments

At the maintainer's request, README numerical results are **transcribed from Tables I, II, and III of the supplied manuscript** (pages 7–8). These are paper-reported values, not newly computed benchmark results. The transcription was checked locally against the supplied PDF. Detailed experiment CSVs and model weights are not part of this publication update.

The paper's tables have different full-model entries: Table I Average uses P=93.7 and mIoU=77.3, whereas Table III Full Model uses P=93.8 and mIoU=77.4. Each table is preserved as printed. Table III A2 uses mIoU=56.3, not the different number in the nearby paragraph. No raw run data is changed to resolve manuscript differences.

The separate September archived checkpoint replay covers 68 held-out scenes and 462 saved progress points, with main-model F1=93.600069% and mIoU=77.371037%. Its scope should not be silently relabeled as the manuscript's larger stated protocol or as repeated independent training runs. Old GRScene-only numbers in `EXPERIMENT_RESULTS.md`, `results/`, and `repro/` remain historical records.

Hardware and robot-side latency are taken from the maintainer's newer report, not the draft's older camera description. The monitoring camera is DJI Osmo Action 4. See `real_robot/README.md` for timing scope and the unresolved total in the source screenshot. No real-world segmentation score is fabricated for the paper's placeholder table.

## Attribution and release scope

New release contributions are authored and committed as **EchO-NN <2579947814@qq.com>**. Existing Git history and third-party copyright statements are not rewritten.

This update publishes source, documentation with the paper's summary tables, and the supplied small media. It does not upload detailed experimental data or model weights. Scene assets, full voxel maps, annotations, private machine credentials, trained checkpoints, external native builds, and the unreleased robot deployment are not bundled. Required paths and dependencies must be supplied by the user; a clean clone is not a self-contained dataset or simulator installation. Existing remote history is preserved, not rewritten to remove historical releases.

The supplied manuscript PDF is used as a reference, not automatically published. Only the explicitly supplied pipeline figure and simulation recording are included as media.
