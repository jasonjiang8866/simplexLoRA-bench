"""Two pre-registered clipping policies; no optimizer or LR changes."""
import math



def parameter_group(name):
    if name == "simplex_importance":
        return "importance"
    if "modules_to_save.default" in name:
        return "head"
    if "lora_magnitude_vector" in name:
        return "magnitude"
    if "lora_A" in name:
        return "adapter_A"
    if "lora_B" in name:
        return "adapter_B"
    return "other"


def clip_gradients(named_parameters, strategy):
    """Rebuild groups every step: Simplex replaces/retires adapter Parameters.

    Importance joins A/B/magnitude in ONE adapter group, not its own group.
    Each group has max_norm=1, norm_type=2, PyTorch's existing 1e-6 epsilon.
    """
    import torch
    if strategy not in ("global", "grouped"):
        raise ValueError("Unknown clipping strategy")
    groups = {"global": []} if strategy == "global" else {"head": [], "adapter": []}
    seen = set()
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        group = parameter_group(name)
        if group == "other" or id(p) in seen or p.grad is None:
            raise ValueError("Unknown, duplicate or missing-gradient clipping parameter")
        seen.add(id(p))
        groups["global" if strategy == "global" else "head" if group == "head" else "adapter"].append(p)
    if any(not params for params in groups.values()):
        raise ValueError("Empty clipping group")
    records = {}
    for group, params in groups.items():
        norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True))
        records[group] = {"pre_norm": norm, "coefficient": min(1.0, 1.0 / (norm + 1e-6)),
                          "max_norm": 1.0, "tensors": len(params)}
    return {"strategy": strategy, "groups": records}


def validate_clip_evidence(gradients, strategy):
    post = gradients["post_clip"]
    if gradients["missing"] or gradients["nonfinite"] or post["missing"] or post["nonfinite"]:
        raise ValueError("Invalid gradient evidence")
    limit = 1.0 if strategy == "global" else math.sqrt(2)
    if not math.isfinite(post["norm"]) or post["norm"] > limit + 1e-5:
        raise ValueError("Gradient clipping bound violated")
    record = gradients.get("clipping")
    if record is None:
        if strategy != "global":
            raise ValueError("Missing grouped clipping evidence")
        return  # Historical global runs remain readable.
    expected = {"global"} if strategy == "global" else {"head", "adapter"}
    if record["strategy"] != strategy or set(record["groups"]) != expected:
        raise ValueError("Clipping strategy/group mismatch")
    pre = gradients["group_norms"]
    after = post["group_norms"]
    for group, item in record["groups"].items():
        keys = list(pre) if group == "global" else ["head"] if group == "head" else ["adapter_A", "adapter_B", "magnitude", "importance"]
        norm = math.sqrt(sum(pre[k] ** 2 for k in keys))
        if not math.isclose(item["pre_norm"], norm, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("Clipping pre-norm mismatch")
        coefficient = min(1.0, 1.0 / (item["pre_norm"] + 1e-6))
        if item["max_norm"] != 1.0 or item["tensors"] < 1 or not math.isclose(item["coefficient"], coefficient, rel_tol=1e-7):
            raise ValueError("Invalid clipping coefficient")
        if math.sqrt(sum(after[k] ** 2 for k in keys)) > 1.00001:
            raise ValueError("Group clipping bound violated")
        for key in keys:
            if not math.isclose(after[key], pre[key] * coefficient, rel_tol=2e-5, abs_tol=1e-7):
                raise ValueError("Post-clip group norm mismatch")


def trainable_signature(model, controller=None):
    """Exact initial tensors, including external importance; no RNG draws."""
    import hashlib
    import torch
    h = hashlib.sha256()
    params = list(model.named_parameters())
    if controller:
        params.append(("simplex_importance", controller.weights))
    for name, p in params:
        if p.requires_grad:
            h.update(f"{name}:{tuple(p.shape)}:{p.dtype}".encode())
            h.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
