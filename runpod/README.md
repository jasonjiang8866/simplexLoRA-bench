# RunPod workflow

The implementation provides both **start inside a Pod** and **create a Pod
from your machine**. No account credentials are embedded. The Actions path
also supports bounded smoke runs and independent Global Volume restore checks.

For the diagnosed/full-matrix path, use the **new immutable image and explicit
dispatch inputs** in [the current experiment section](#current-formal-dispatch).
The original smoke image below remains historical evidence and is still used
by the separate legacy smoke workflow.

## Historical first published image (2026-11-01)

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
availability from the verification host. Actual RunPod pull/startup and CUDA
smoke evidence are recorded separately below.

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
CUDA/int8 compatibility result or a GPU benchmark result.** The subsequent
[2026-11-01 smoke record](../docs/results/runpod-smoke-41324992190.md) documents
actual GPU execution and Global Volume persistence. That historical release
did not complete the full 500-step performance benchmark; the current formal
dispatch is recorded separately below.

## 1. Build and publish another revision

Run **Publish RunPod image** on the desired branch in GitHub Actions. The job
builds the CUDA 12.8 / PyTorch 2.9.0 image, publishes a source commit tag and
uploads `image.txt` with the immutable digest. Keep the GHCR package public
for the supplied launcher, or configure registry authentication through the
RunPod console if using a private image.

Alternatively, build/push locally:

```bash
docker build --build-arg SOURCE_SHA="$(git rev-parse HEAD)" -f runpod/Dockerfile -t ghcr.io/jasonjiang8866/simplexlora-bench:experiment .
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

If GPU smoke succeeded but the later restore phase failed, dispatch with
`confirm_paid_smoke=true` and `restore_from_run=<original Actions run ID>`.
This retrieves attempt 1's artifact, checks the completed GPU result, archive
SHA-256, volume identity and confirmed GPU cleanup, then rents **only a CPU**.
The source artifact must still be available (14-day retention). The dataset
keeps its original run ID; cleanup ownership belongs to the new Actions run.
The small CPU flavor uses a 10 GB writable disk; the GPU uses 40 GB.

The account budget check uses GraphQL; resource inventory and lifecycle use
RunPod REST v2. Result requests identify the client with an explicit User-Agent.
The result endpoint requires a per-attempt token; provider errors are logged
only as fixed diagnostic categories. `Inspect managed RunPod compute` provides
read-only checks of this repository's managed Pods.

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

For the console/regional-volume launcher, **the training timeout and container
exit do not stop Pod billing.**
Stop/terminate the Pod through RunPod after completion or failure; retain the
network volume for results. That launcher does not automatically remove compute;
the Actions smoke controller described above does remove its own test Pods.

Restarting with the same run ID verifies hashes and skips completed runs.
Incomplete attempts are archived and restarted from zero. A changed image/source
or config must use a new run ID. Do not run concurrent Pods against the same ID.

## Gradient diagnosis and full-matrix Actions (new)

`RunPod gradient diagnosis and full benchmark` accepts an immutable image digest
and its exact source commit. It checks out that commit and verifies both the
baked `/opt/source-sha.txt` and the training source hash on the GPU and CPU.
Publish the changed image first; the historical digest above has no new probes.

Dispatch without `confirm_paid_experiment` first: this only checks inventory,
current MIG/CPU rates, the exact Global Volume ID, budget headroom, and writes
`plan.json`. No paid creation occurs. The GPU remains the specified PRO 6000
Blackwell MIG 24GB with a $0.69/hour ceiling; there is no fallback GPU.

The `diagnostic` profile is the first 50 optimizer steps of the 500-step schedule,
with the **full** 8192/1024/2048 data subsets, batch 8 and length 128. It uses all
three paired seeds, includes rank changes at 10/20/30 and evaluations at 25/50.
Both diagnostic and benchmark pin the model/data revisions used by the old
smoke. No learning rate, initialization, loss, clipping or spike threshold is
changed. Group norms, analytic head-gradient agreement, pooling/padding checks,
batch hashes, post-clip norms, logits, hidden norms and optimizer/rank transitions
are recorded in each step. Monitoring is charged to both methods' timing.

The full profile requires all six **500-step** runs; it cannot pass with smoke
results. Maximum result export is 1 GiB (512 MiB file payload plus archive
headroom), versus the historical smoke's 256 MiB. CPU restoration retains a
10 GB writable disk and a 15-minute deadline. GitHub experiment artifacts are
kept for 90 days; completion markers and per-file SHA-256 remain on Global
Volume, written after the archive. Training/cache remain local `/workspace`.

Supply an explicit `gpu_minutes` and `max_compute_usd`. The cap must cover GPU
time plus five minutes cleanup reserve, and CPU 15m plus five minutes cleanup
reserve. Example: diagnosis 45m requires $0.6417 at rate ceilings; the proposed
$0.65 limit is **not** a full-benchmark authorization. The cap is checked against
account balance and $500 monthly headroom, not substituted with the old $1
reserve. Storage/request charges remain separate. Provider or runner outages
can delay deletion beyond the reserved cleanup window; this is a controller
deadline, not a provider-guaranteed billing cutoff.

After reviewing diagnostic evidence, commit `docs/results/diagnosis-review.json`
with `decision`, `image`, `source_hash`, `diagnostic_run`, and a rationale. Its
workflow publishes `diagnosis-review`. A formal dispatch requires the artifact's
run ID and matching image/source. CPU-only recovery uses `restore_from_run` and
the original profile inputs; it verifies the prior GPU archive and confirmed
cleanup, and creates no replacement GPU. This recovers a **completed matrix**;
partial-matrix cross-Pod resume is not implemented. The CLI's local resume does
not imply that incomplete local Pod data survives deletion.

Analyze downloaded step evidence with:

```bash
python runpod/report_experiment.py smoke-evidence.tar --output analysis.json
python runpod/render_report.py analysis.json --output tables.md
# Optional local analysis dependency: matplotlib (validated with 3.10.8).
python runpod/plot_experiment.py analysis.json --output curves.svg
```

The archive retains the historical filename `smoke-evidence.tar` because it
reuses the verified transport; `plan.json`, acceptance `performance_benchmark`,
resolved config and each run's step count distinguish the actual experiment.

### Verified diagnostic image and dry-run — 2026-11-02 SGT

- Source: `b5d40812c9ba7eb1a8ef8c28d4b9e0e262367753` (PR #2, merged).
- Image: `ghcr.io/jasonjiang8866/simplexlora-bench@sha256:2d630d23e069ed5b486520dbd303c3c35a7cc8c56ec79d269c710819adf7fa08`.
- Training source hash: `2f1b91c4b02e5d0b937edfd962ef941cb6f1f1497cf3470eba93d4be971734b9`.
- [Image publication](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41330245339): success; 29 CPU tests and the CPU PEFT gate executed inside the new image build.
- The public registry tag and computed OCI index hash match the digest above;
  downloaded `image.txt` agrees. The historical smoke image is unchanged.
- [Experiment dry-run](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41330661456): success. `diagnostic`, 45 GPU minutes, $0.65 proposed compute cap, $0.641667 ceiling including cleanup reserve. This preparation run created no compute.
- [Inventory check](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41330478365): zero managed Pods. Preflight reported zero account Pods, MIG $0.69/h with EUR-IS-2/US-PA-1/US-WA-1 availability, CPU cpu3c AP-JP-1 $0.06/h, and the expected Global Volume.

The user subsequently approved a total $10 session budget. [Diagnostic run 41331245400](../docs/results/gradient-diagnosis-41331245400.md) completed successfully, including independent CPU restoration and confirmed cleanup, at estimated compute cost $0.12654. The [full six-run benchmark](../docs/results/benchmark-41333468131.md) then completed using the same image, a 75-minute GPU deadline and $1 compute cap. Its compute estimate is $0.431960; diagnostic plus formal total is $0.558499, excluding existing storage/request charges.

### Current formal dispatch

**Verified complete:** six 500-step workers, CUDA/int8 gate and CPU restore;
both Pods deleted. [Final inspection](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41336870451)
confirmed zero managed Pods. Global Volume data remains at
`/global/simplexlora-bench/simplex-41333468131-1/`.

The formal read-only [dry-run 41333320173](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41333320173)
passed with zero account Pods, MIG $0.69/h and cpu3c $0.06/h. Paid formal
[run 41333468131](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41333468131)
was dispatched with these exact inputs (new deployments must recheck inventory):

| Input | Value |
|---|---|
| mode | `benchmark` |
| image | `ghcr.io/jasonjiang8866/simplexlora-bench@sha256:2d630d23e069ed5b486520dbd303c3c35a7cc8c56ec79d269c710819adf7fa08` |
| image_source_sha | `b5d40812c9ba7eb1a8ef8c28d4b9e0e262367753` |
| gpu_minutes | `75` |
| max_compute_usd | `1` |
| confirm_paid_experiment | `true` (`false` for read-only dry-run) |
| diagnosis_review_run | `41333037273` (review publication run, not the diagnostic run ID) |
| restore_from_run | empty for a new matrix |

The controller checks out the image's exact source commit even when main has
later analysis/documentation commits. This prevents a moving main from changing
the training or controller during six paired runs. The source/image mismatch
checks are not relaxed for later report-only changes.

For CPU-only recovery of this completed matrix, use the same inputs with
`restore_from_run=41333468131`; no replacement GPU is created. Recovery requires
the retained original Actions artifact and confirmed original GPU deletion.
Do not rerun the whole paid job merely to retry a failed CPU restoration.

Both `provision-*.json` files must show `cleanup_confirmed: true`. After the job,
run **Inspect managed RunPod compute** to independently verify that managed Pod
count is zero. Preserve the Global Volume and unrelated resources.


## Three-method DoRA profiles

Select `diagnostic-dora` (nine × 50 prefix steps) or `benchmark-dora`
(nine × 500 steps) in `RunPod gradient diagnosis and full benchmark`.
The historical `diagnostic` / `benchmark` choices remain two-method profiles.
Use the new image digest and exact source SHA from publication; the historical
image cannot execute DoRA. See [current preparation evidence](../docs/results/dora-preflight.md).

First dispatch with paid confirmation **false**. The dry-run creates no compute.
Proposed diagnostic limits: 45 GPU minutes, 15 CPU minutes, $0.65 compute cap
including five-minute cleanup reserves per phase. Existing controller ceilings
are MIG $0.69/hour and CPU $0.20/hour; actual cpu3c was previously $0.06/hour
and must be rechecked. No more expensive GPU fallback is permitted. This plan
is not budget authorization; request any new session allocation before creation.
Use diagnosis timings to select the formal deadline/cap, rather than assuming
DoRA costs the same as LoRA. The completed run below now supplies measured costs.

For formal DoRA dispatch, the reviewed diagnosis artifact must bind the image,
source hash, diagnostic Actions run, `methods: ["lora", "simplex", "dora"]`,
`completed_runs: 9` and `diagnostic_steps: 50`. A two-method diagnosis cannot
unlock a three-method formal run. Keep the old review record; publish the new
review in `docs/results/dora-diagnosis-review.json`, then dispatch
`Publish reviewed gradient decision` with `experiment=dora` once the evidence exists.

The result archive remains bounded to 1 GiB (512 MiB file payload), 1000 members,
90-day Actions retention, and a 10 GB CPU writable disk. Nine runs plus added
telemetry are expected to fit well within this bound; actual sizes/file counts
must be measured, never assumed to be the historical 110. Training/cache stay
local; per-file SHA-256 is sealed before the completion marker is written last.
A new run ID preserves every historical Global Volume directory.

GPU removal precedes CPU-only cross-region recovery; recovery needs the original
profile/image/source and confirmed prior GPU cleanup. It does not rent a GPU or
require another diagnosis review for an already-completed formal archive.
The normal `finally` plus Actions `always()` ownership checks remain in force.

### Completed three-method run — 2026-11-02 SGT

[Diagnosis 41396288500](../docs/results/dora-diagnosis-41396288500.md) passed
nine × 50 steps, then [formal 41399541318](../docs/results/dora-benchmark-41399541318.md)
passed nine × 500 steps. Both used source
`fba99733fd746af43fe7574e9cbcddb775737162` and immutable image
`ghcr.io/jasonjiang8866/simplexlora-bench@sha256:3af1106bf5798af9643cce4792f50cea4af832c68a9e803b87dedec9e8a77b71`.
The review publication run was `41399432092`. The measured diagnosis forecast
selected formal limits of 90 GPU minutes / 15 CPU minutes / $1.20 phase cap.
Live rates stayed at MIG $0.69/hour and cpu3c $0.06/hour.

The formal export is `/global/simplexlora-bench/simplex-41399541318-1` on the
existing volume `cmvwmviz9000007jteci4ajn8`. After GPU deletion in US-PA-1, a new
CPU in EU-CZ-1 verified all 170 files, 163,209,130 payload bytes and the same
summary, then was deleted. [Final inspection 41406335557](https://github.com/jasonjiang8866/simplexLoRA-bench/actions/runs/41406335557)
confirmed zero managed Pods. Diagnosis plus formal compute estimate was
$0.828779 within the user's $4.50 authorization; storage/request fees are extra.
The report links the 90-day Actions artifact and records every archive/image/hash
and lifecycle identity. Historical result directories and archives remain intact.
