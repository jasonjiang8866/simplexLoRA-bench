# RunPod workflow

The implementation provides both **start inside a Pod** and **create a Pod
from your machine**. No account credentials are embedded. No Pod has been
created by the repository setup.

## Published image (2026-11-01)

The first image was published by the existing manual
[Publish RunPod image run](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41320152026)
(completed successfully). The
[GHCR package](https://github.com/jasonjiang8866/simplexLoRA-bench/pkgs/container/simplexlora-bench)
is **Public**; registry credentials are not required to pull it.

| Item | Verified value |
|---|---|
| Source main SHA at publication | `dd2913517f13061c97425873e822a02f4ee871b1` |
| Tag | `ghcr.io/jasonjiang8866/simplexlora-bench:sha-dd2913517f13061c97425873e822a02f4ee871b1` |
| Immutable image index digest | `sha256:cb5ce33250b8ced35892adbbcdc8c531eee67d5bab99192907a8fba4d701d342` |
| Platform | `linux/amd64` |
| Platform manifest digest | `sha256:cafcab6d4df1ea3d5909550f7c7fb5b03d099fabdd83b49a952e2462bebe4b4b` |
| Workflow artifact | `runpod-image-reference` containing `image.txt` |

Use this exact reference for deployment and the launcher examples below:

```bash
export BENCH_IMAGE_DIGEST='ghcr.io/jasonjiang8866/simplexlora-bench@sha256:cb5ce33250b8ced35892adbbcdc8c531eee67d5bab99192907a8fba4d701d342'
```

The tag, workflow output and registry manifest digest agree. The image contains
the source SHA above; subsequent documentation commits do not change its code.
Anonymous OCI registry verification retrieved the index by tag and digest,
the `linux/amd64` manifest, image configuration and all eight layer blobs.
Each layer's byte count and SHA-256 matched its manifest descriptor. The image
configuration confirms `/bin/bash /opt/bench/runpod/start.sh` as the entrypoint
and `/opt/bench` as the working directory. This verifies public registry
availability from the verification host; it is not a pull or startup test on
an actual RunPod node.

The Dockerfile uses `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime`, resolved
in this build to
`sha256:f0ca81b440e252399d9954a45b616ee2540959466aacf3dfc3f856691eee66e8`.
Direct runtime dependencies are pinned: torch 2.9.0, transformers 4.57.1,
peft 0.17.1, accelerate 1.11.0, datasets 4.3.0, huggingface-hub 0.36.0,
scikit-learn 1.7.2, numpy 2.2.6, safetensors 0.6.2 and bitsandbytes 0.48.1;
pytest is pinned to 8.4.2. Transitive dependencies are not fully locked.
The workflow has `contents: read` and `packages: write` permissions and uses
`GITHUB_TOKEN` for GHCR publication.

The Dockerfile includes `pip check`, non-CUDA tests and CPU PEFT compatibility
checks. This publication reused the cached build layer containing those steps;
it did not freshly execute those checks. **A successful image build is not a
CUDA/int8 compatibility result or a GPU benchmark result.** No paid Pod was
created for this release; actual RunPod startup, CUDA/int8 compatibility and
the six GPU training runs remain to be validated.

## 1. Build and publish another revision

Run **Publish RunPod image** on the desired branch in GitHub Actions. The job
builds the CUDA 12.8 / PyTorch 2.9.0 image, publishes a source commit tag and
uploads `image.txt` with the immutable digest. Keep the GHCR package public
for the supplied launcher, or configure registry authentication through the
RunPod console if using a private image.

Alternatively, build/push locally:

```bash
docker build -f runpod/Dockerfile -t ghcr.io/jasonjiang8866/simplexlora-bench:experiment .
docker push ghcr.io/jasonjiang8866/simplexlora-bench:experiment
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/jasonjiang8866/simplexlora-bench:experiment
```

Use the returned digest for a new build instead of the first-release digest.
The image includes the source revision's code; startup does not clone a mutable
branch or install extra packages. A dedicated venv isolates project dependencies
from the container's user environment while reusing its PyTorch installation.

## 2. Start on RunPod

### Actions: smoke and Global Volume persistence

`RunPod smoke and Global Volume` uses the repository's `RUNPOD_API_KEY` Actions
secret. Pushes run a read-only preflight; manually dispatch with
`confirm_paid_smoke=true` to rent compute. It uses the published digest above,
without replacing the benchmark code or installing packages in the Pod.

The controller requires the existing Global Volume named
`spt-1436489970-step90-inputs-v1`; it creates no storage. One PRO 6000 Blackwell
MIG 1g.24gb runs the original smoke entrypoint, then a separate CPU Pod in a
different data center restores and verifies the completed results. The mount
is `/global`; this project's new data lives under
`/global/simplexlora-bench/simplex-<Actions run>-<attempt>/`.
Training and the Hugging Face cache stay on local `/workspace` because Global
Volume object storage must not be assumed to support POSIX locks or atomic
rename. Completed results are published as `evidence.tar` with file hashes,
an 8 MiB access probe, and a completion marker written last. The CPU reader
verifies every file and regenerates the summary from restored artifacts.

The GPU rate cap is $0.69/hour for at most 45 minutes, followed by a CPU rate
cap of $0.20/hour for at most 15 minutes. Preflight checks a $1 reserve and
headroom under the $500 monthly budget. Storage/request charges are additional.
The account credential remains on the Actions runner; only an ephemeral result
token reaches the Pod. The controller cleans up compute in `finally`, with a
second `always()` cleanup step, and checks that the Pods disappeared. A hard
runner failure can still require manual cleanup; container exit alone does not
stop billing. Global Volume contents and unrelated Pods are retained.

Download the `simplex-runpod-<run>-<attempt>` artifact for preflight, provision,
result, cleanup and acceptance evidence. A passed preflight is not a smoke
result; a smoke result is not the full performance benchmark.

### Console or regional-volume launcher

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
are pipeline diagnostics only. After resolving the model/dataset revisions,
the matrix runner executes `python -m simplexbench.compat --cuda` before any
training worker. A failing gate blocks all six workers in either mode.

The release was checked with the real digest using the following dry-run:

```bash
python runpod/launch.py \
  --image "$BENCH_IMAGE_DIGEST" \
  --network-volume DRY_RUN_VOLUME \
  --data-center DRY_RUN_DATA_CENTER \
  --run-id simplex-smoke-image-validation --mode smoke
```

The output had `dry_run: true`, the exact digest, one GPU and
`BENCH_MODE=smoke`. The placement IDs above are placeholders, not existing
resources. No create request was sent. A mocked subprocess control-flow check
also verified that the CUDA gate runs before the six workers and that gate
failure prevents every worker from starting; this did not execute CUDA.

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
