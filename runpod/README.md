# RunPod workflow

The implementation provides both **start inside a Pod** and **create a Pod
from your machine**. No account credentials are embedded. No Pod has been
created by the repository setup.

## 1. Build and publish

CI builds the CUDA 12.8 / PyTorch 2.9.0 image and runs CPU regression/compatibility
inside it. The current environment cannot run Docker or CUDA.

Once the workflow is on the default branch, run **Publish RunPod image** in
GitHub Actions. The job publishes a commit tag and uploads `image.txt` with
the immutable digest. Make the GHCR package public for the launcher below,
or configure registry authentication through the RunPod console.

Alternatively, build/push locally:

```bash
docker build -f runpod/Dockerfile -t ghcr.io/jasonjiang8866/simplexlora-bench:experiment .
docker push ghcr.io/jasonjiang8866/simplexlora-bench:experiment
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/jasonjiang8866/simplexlora-bench:experiment
```

Use the returned digest; no usable image digest is invented in this repository.
The image includes the source revision's code; startup does not clone a mutable
branch or install extra packages. A dedicated venv isolates project dependencies
from the container's user environment while reusing its PyTorch installation.

## 2. Start on RunPod

Choose a single CUDA GPU, attach an existing network volume at `/workspace`
and select the image digest. Set:

| Environment variable | Value |
|---|---|
| BENCH_MODE | smoke first, then benchmark |
| BENCH_RUN_ID | unique name such as simplex-smoke-001 |
| BENCH_TIMEOUT_SECONDS | 14400 by default |

The image entrypoint starts automatically:
CUDA int8 compatibility gate → six smoke/full runs → verified summary.
Results and Hugging Face cache persist under `/workspace/results/<run-id>`
and `/workspace/huggingface`.

The smoke preset uses three seeds × two methods × eight steps. Its F1 values
are pipeline diagnostics only. A failing gate blocks the full experiment.

To launch via API from your own machine, first preview the exact payload:

```bash
python runpod/launch.py \
  --image "$BENCH_IMAGE_DIGEST" \
  --network-volume "$RUNPOD_VOLUME_ID" \
  --data-center "$RUNPOD_DATA_CENTER" \
  --run-id simplex-smoke-001 --mode smoke
```

Use a volume and data center that match. The default GPU is RTX 5090; select
another available GPU with `--gpu`. Inspect live price/availability in RunPod.
To create the paid Pod, set `RUNPOD_API_KEY` locally and add `--launch`.
Only one POST is issued. On an ambiguous timeout, check the console before
retrying to avoid duplicate billed Pods. The API key is never sent to the Pod.
The supplied launcher expects a public image and ungated model.

This uses **REST API v2**, `POST https://api.runpod.io/v2/pods`; v1 is being
retired. Payload fields were checked against RunPod's official
[OpenAPI schema](https://github.com/runpod/docs/blob/main/api-reference-v2/openapi.json)
and [Python SDK](https://github.com/runpod/runpod-python/blob/main/runpod/api/ctl_commands.py).
The create response is provisioning status, not benchmark completion.

## 3. Inspect and stop

Inspect `console.log`, `exit-code.txt`, `compatibility.json` and
`summary.md/json` on the network volume using a volume-attached management
Pod or the RunPod console. The API launcher does not expose a public service
or configure SSH. Download results before deleting the volume.

**The training timeout and container exit do not stop Pod billing.**
Stop/terminate the Pod through RunPod after completion or failure; retain the
network volume for results. No automatic destructive cleanup is performed.

Restarting with the same run ID verifies hashes and skips completed runs.
Incomplete attempts are archived and restarted from zero. A changed image/source
or config must use a new run ID. Do not run concurrent Pods against the same ID.
