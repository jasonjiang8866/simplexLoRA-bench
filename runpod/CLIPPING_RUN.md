# Clipping GPU execution proposal — 2026-11-03 (SGT)

Status: GPU integration and browser Actions dispatch authorized on 2026-11-03;
US$5 total experiment budget approved. Image/preflight/GPU results pending.
Approved training code is PR #4 at `713e5ea034b678c929f49690ff96bf4fe86abac5`.
This file records the reviewed integration and authorized execution scope.

## Integration

- Add manual `diagnostic-clipping` and `benchmark-clipping` workflow choices.
- Route the immutable image entrypoint to `simplexbench.ablation` for these modes.
- Require 18 distinct method/seed/strategy runs, nine pairs, the registered
  alternating order, both policies' telemetry, and the real CUDA/int8 gate.
- On GPU and CPU restoration, regenerate the nested summary and require an
  unchanged hash, exact config and source identity; deep initialization, batches,
  gradient bounds, rank budgets and checkpoint checks remain in the summarizer.
- Bind formal authorization evidence to the same image/source, 18 completed
  diagnostic workers, both strategies, diagnostic config hash, successful pairing,
  independent CPU recovery and confirmed cleanup. Old DoRA reviews cannot pass.
- Increase only clipping archive limit to 2 GiB (1 GiB file payload); keep the
  1,000-member bound, 10 GB CPU disk and 90-day Actions artifact retention.
- Preserve manual paid confirmation, concurrency, deadlines, finally/always
  cleanup and existing Global Volume. No auto-paid push trigger or GPU fallback.

Validation: 44 CPU tests passed (one CUDA test deselected); 21 lifecycle tests
passed. The CPU nested-archive test uses synthetic gate metadata; it is not CUDA
verification. It extracts all 18 small fixture runs, regenerates identical
summary/file hashes, and rejects a corrupted restored steps file and wrong
expected config. YAML parsed with manual-paid guards checked; `bash -n` and
`git diff --check` passed. Integration Docker/CUDA validation is still pending.

## Approved total allocation: US$5.00

One NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb; no other GPU fallback.
The official public price inspected on 2026-11-03 is $0.69/hour:
https://www.runpod.io/pricing (page says updated 2026-10-28).
Actual account catalog price/availability, region and volume still require a
read-only preflight, and must remain within this ceiling before creation.

| Phase | Matrix | GPU deadline | CPU deadline | Compute ceiling including cleanup | Dispatch compute cap |
|---|---|---:|---:|---:|---:|
| Diagnosis | 18 x 50 prefix steps | 45 minutes | 15 minutes | $0.641667 | $0.65 |
| Formal | 18 x 500 steps from zero | 150 minutes | 15 minutes | $1.849167 | $1.85 |
| Total | 9,900 steps | 195 minutes | 30 minutes | $2.490834 | $2.50 |

Each phase includes an additional five-minute cleanup allowance for each GPU/CPU
Pod; CPU rate ceiling is $0.20/hour. The $2.50 unallocated remainder covers this
experiment's temporary disk, incremental volume/request/transfer charges,
applicable tax and any separately reconciled recovery within the same $5 total.
Do not automatically rerun a GPU matrix. Compute cap inputs do NOT include storage
and do not implement an account-wide total cap: track aggregate actual/estimated
cost against $5 before each dispatch. A runner/provider outage can delay cleanup;
the deadline is not a provider-guaranteed billing cutoff.

Container disk is publicly listed at $0.10/GB/month, with no Pod ingress/egress
fees: https://docs.runpod.io/pods/pricing . Forty GB on GPU and ten GB on CPU
would cost about $0.02 over these deadlines (30-day-month estimate). Global Volume
incremental pricing/request fees were not confirmed in this session; verify them
before allocating, reserve them from $2.50, and do not equate global storage with
regional network-volume pricing. Preserve the existing volume and historical
results. Retained storage can continue accruing after compute is deleted; disclose
that ongoing cost separately in the final result. Existing unrelated storage
billing is not charged against this experiment's incremental allocation.

Estimated observed GPU time from the prior nine-run matrix is roughly 2.5–3 hours
including overhead for two new policies; it is not a measured clipping result.
If diagnosis projects the full run cannot fit 150 minutes / remaining funds,
stop at the measured diagnosis and revise the proposal before further compute.

## Launch sequence after approval

1. Push this reviewed integration to PR #4; wait for Python 3.11/3.12 and Docker
   checks. No merge is required for an existing workflow dispatched on this branch.
2. Publish RunPod image on `feat/paired-gradient-clipping`. Record the actual full
   source SHA and immutable digest from the successful build; verify provenance.
   Never use the old DoRA image or the training-only PR SHA for this integration.
3. Use the existing `runpod-experiment.yml` on that branch with:
   `mode=diagnostic-clipping`, `image=<verified new digest>`,
   `image_source_sha=<verified full build SHA>`, `gpu_minutes=45`,
   `max_compute_usd=0.65`, `confirm_paid_experiment=false`,
   `restore_from_run=''`, `diagnosis_review_run=''`.
4. Inspect plan/catalog/owned Pods and exact existing Global Volume
   `cmvwmviz9000007jteci4ajn8`, account headroom and prices. Do not create/resize a
   volume or reuse an occupied run identity. Unknown inventory blocks launch.
5. Only after an explicit new $5 budget authorization and passing preflight,
   dispatch the same inputs with paid confirmation true. Capture run ID/URL once.
6. Retrieve and independently verify diagnosis; measure timing, archive size and
   full-run cost. Create a factual `docs/results/clipping-diagnosis-review.json`
   and publish via `diagnosis-review.yml` with `experiment=clipping`.
7. Formal inputs: `mode=benchmark-clipping`, same image/source,
   `gpu_minutes=150`, `max_compute_usd=1.85`, paid confirmation true,
   `diagnosis_review_run=<actual review publication run>`, no restore ID.
   Run another read-only preflight first. Do not retune based on diagnostic F1.
8. Recover complete archives on a fresh CPU in a different region, regenerate
   summaries, download artifacts, and verify temporary compute deletion with the
   independent inspect workflow. If CPU recovery alone fails, use
   `restore_from_run` with identical profile/image and original GPU deletion proof;
   reconcile remaining total funds first. Do not recreate GPU merely for recovery.

The GitHub connector in this session has no workflow-dispatch tool and `gh` is
not installed. Browser triggering was explicitly authorized on 2026-11-03. Use the exact repository Actions UI
for image publication, dry-run, experiment, review publication and inspection.
No secret is copied out of GitHub Actions. No merge, repository settings, secret
changes or persistent volume deletion is included in this proposal.
