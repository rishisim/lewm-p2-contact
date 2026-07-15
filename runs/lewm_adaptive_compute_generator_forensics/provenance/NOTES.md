# Bounded Cube generator provenance audit

## Result

The public evidence narrows the likely generator substantially, but does not prove an exact source revision. The strongest bounded historical candidate is `stable-worldmodel` commit `a265229cb29688715651cbd831a3b4c10b8f98b4` paired with OGBench v1.2.1. That commit is titled **“fix cube script”** and its checked-in collector/config specify `swm/OGBCube-v0`, single-cube collection, `ExpertPolicy`, 10,000 trajectories, a 200-step horizon, 224×224 multiview rendering, seed 3072, and no termination at goal. These are unusually close to the released artifact's independently established dimensions.

This is still a candidate, not recovered provenance. The Hugging Face card is only 143 bytes and contains no generator command, package lock, seed, commit, or render metadata. The public collector names `cube_single_multiview_expert`, not the released HDF5 archive. The stable-worldmodel dependency on OGBench is unpinned. No source/config identifiers are embedded in the HDF5 metadata according to the prior immutable structural audit.

## Direct evidence

- [Hugging Face repository](https://huggingface.co/datasets/quentinll/lewm-cube) identifies the artifact as the official OGBench-Cube dataset used in LeWorldModel. Its API head is `02a19a67...`; all six visible commits are initial/upload/README commits on March 26–27, 2026.
- [Hugging Face tree API](https://huggingface.co/api/datasets/quentinll/lewm-cube/tree/main?recursive=true&expand=true) reports archive size 46,184,624,478 bytes, LFS SHA-256 `3725d6a0...`, Xet hash `ffe07ef6...`, and upload revision `f6fe4695...`.
- [stable-worldmodel collector at a265229](https://github.com/galilai-group/stable-worldmodel/blob/a265229cb29688715651cbd831a3b4c10b8f98b4/scripts/data/collect_cube.py) and its [Hydra config](https://github.com/galilai-group/stable-worldmodel/blob/a265229cb29688715651cbd831a3b4c10b8f98b4/scripts/data/config/ogb.yaml) provide the closest public generation recipe found.
- [OGBench v1.2.1](https://github.com/seohongpark/ogbench/releases/tag/v1.2.1), commit `1d414099...`, predates the artifact. Its [official commands](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/data_gen_scripts/commands.sh#L111-L120) use `dataset_type=play` for Cube single, while [generate_manipspace.py](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/data_gen_scripts/generate_manipspace.py) maps play to `CubePlanOracle` and noisy to `CubeMarkovOracle`.
- The official OGBench recipe itself is not the released recipe: it specifies 1,000 episodes and `max_episode_steps=1001`, versus the released artifact's 10,000 episodes and 200 actions.
- The installed environment is stable-worldmodel 0.1.0, OGBench 1.2.1, MuJoCo 3.10.0, dm-control 1.0.43, Gymnasium 1.3.0, NumPy 2.2.6, h5py 3.16.0, and hdf5plugin 6.0.0. Installed Cube source is pinned by byte hash in `provenance.json` because a package version alone does not prove a git revision.
- The local LeWM checkout points to [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm). Its initial March 2026 code treats the HDF5 as downloaded input and asks users to install `stable-worldmodel[train,env]`; no LeWM-side Cube generator or exact environment lock was found.

## Inferences and bounded candidates

1. `stable-worldmodel@a265229` + OGBench v1.2.1 is the primary candidate. The matching 10,000×200 configuration and single-cube collector are direct evidence; attribution to the released archive is an inference.
2. `stable-worldmodel@0.0.4` (`089efbb8...`) + OGBench v1.2.1 is retained only as a weaker timestamp candidate because it was the latest tag before the extracted file's preserved February 18 mtime. Archive-preserved mtime is not a reliable generation timestamp.

No third candidate was added. There is no direct evidence supporting a broader revision sweep. Candidate execution, if the mechanics layer requires it, should use the already frozen transitions and remain isolated from the installed environment.

## Audit constraints

This provenance task did not open HDF5 row data, access a V3 cache, execute the environment, modify installed packages, contact authors, or launch any rollout. It used filesystem metadata, existing immutable manifests, local git/package metadata, GitHub primary-source files/releases/APIs, and Hugging Face primary-source APIs only.
