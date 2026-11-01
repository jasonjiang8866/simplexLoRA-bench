# Gradient investigation: implementation and evidence plan

Baseline main: `c0ac29d5d7a128fbc5133923103d16dc5ded7c5c`.
The old GPU artifact ZIP was retrieved and independently hashed:
`f27648e277bc4f5df443d1c9c8befd5f5f480f7093998056a669ee2e9cfdc0ac`.
All 48 step records were read: 46 >100 flags, no missing/nonfinite gradients.
For seeds 17/42/123 respectively, both methods have exactly matching first-step
(loss, norm): (1.384,185.63), (12.609,203.62), (5.206,238.10), rounded here.
This rules out a rank-update origin for the **initial** large gradients, but
does not establish their source or longer-run stability.

Code inspection finds batch-mean sequence-classification cross entropy,
integer labels, a randomly initialized trainable FP32 `score` head, frozen int8
base, FP32 adapters, and joint clipping at 1 after logging. A-gradient zeros
at B=0 initialization are expected. Simplex reinitializes A/B at rank updates,
drops replaced adapter moments and retains classifier moments. These are
method choices already documented, not silently repaired in this change.

New probes measure the norm decomposition and analytically reconstruct the
head gradient `(softmax(logits)-onehot(labels)).T @ pooled_hidden / batch_size`.
They verify pooling against the last attention-masked token, equality with
model logits, mean-CE loss, post-clip norm <=1, optimizer parameter coverage,
and retention of classifier state at every rank update. Groupwise norms and
batch/initial-head hashes support paired comparisons. The 50-step diagnostic
uses the exact first 50 learning rates of the 500-step schedule.

No threshold or training hyperparameter is adjusted. GPU diagnosis and formal
500-step results are pending; CPU tests and image builds do not replace them.
