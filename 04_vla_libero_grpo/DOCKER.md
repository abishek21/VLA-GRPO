# Faster next time — Docker image for RunPod

Setup took ~2 h of dependency wrangling. This bakes the *exact* working stack into
an image so next pod boots ready-to-run in minutes.

## Key fact: you do NOT need a GPU (or NVIDIA) to BUILD the image
`docker build` only *installs* packages — it never runs CUDA. The GPU is used only
at **runtime on RunPod**. So a CPU-only laptop, GitHub Actions, or any CPU VM can
build it. (The image ends up ~15–20 GB, so make sure you have disk + bandwidth.)

Files here:
- `Dockerfile` — the whole M1 environment, baked in.
- `build_and_push.sh` — build & push from any Docker host.
- `../.github/workflows/build-oft-image.yml` — build in CI, zero local resources.

---

## Option B (recommended): GitHub Actions — nothing runs on your laptop
1. Push this repo to GitHub.
2. Actions tab → **build-oft-image** → **Run workflow**.
3. It builds `linux/amd64` and pushes to
   `ghcr.io/<you>/openvla-oft-libero:cu128`.
4. GitHub → Packages → that image → **make public** (so RunPod can pull it).

## Option A: build on your laptop (no GPU needed)
Install Docker Desktop, then:
```bash
cd 04_vla_libero_grpo
export IMAGE=docker.io/<your-dockerhub-user>/openvla-oft-libero:cu128
./build_and_push.sh
```
- **Apple Silicon (M1/M2/M3)?** The script already passes `--platform linux/amd64`
  so the image runs on RunPod's amd64 GPUs. Cross-build is slower (emulated) but works.
- **x86 laptop?** Native, faster.

## Option C: cheap CPU cloud VM
Any small VM with Docker (e.g. a CPU-only cloud instance). Native amd64 = fastest
build. Same commands as Option A.

---

## Use it on RunPod
New Pod → **Edit Template / Custom** → **Container Image** =
`ghcr.io/<you>/openvla-oft-libero:cu128` (or your Docker Hub tag).
Pick a Blackwell/Ada GPU. Then inside the pod:
```bash
export MUJOCO_GL=egl
cd /opt/openvla-oft
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial --center_crop True --num_trials_per_task 5
```
Everything (torch cu128, LIBERO, EGL libs, config, patches) is already in place.

### Optional: also bake in the 14 GB checkpoint
By default the checkpoint downloads on first run (into `HF_HOME`). To bake it into
the image (bigger image, but zero first-run download), add before `CMD` in the
Dockerfile:
```dockerfile
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial')"
```

## Notes
- Base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` already contains the
  Blackwell-capable torch; the Dockerfile only adds the app stack on top.
- If that base tag is unavailable, switch to
  `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` and uncomment the pip cu128 block
  in the Dockerfile.
- `requirements-frozen.txt` (if generated) captures exact versions for auditing;
  the Dockerfile pins the critical ones inline.
