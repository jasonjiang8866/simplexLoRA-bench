"""Independent implementation of arXiv:2610.37206, section 3.1.

Uses stock PEFT adapters, never changes the frozen quantized base. No upstream
fork code is vendored. See docs/METHOD.md for paper/code differences.
"""
import math

import torch
from peft.tuners.lora import LoraLayer


def project_simplex(values, mass):
    """Euclidean projection onto nonnegative vectors summing to mass."""
    if values.ndim != 1 or not values.numel() or mass <= 0:
        raise ValueError("A nonempty vector and positive mass are required")
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite importance weights")
    v = values.double()
    ordered = torch.sort(v, descending=True).values
    cumulative = ordered.cumsum(0) - mass
    indices = torch.arange(1, v.numel() + 1, device=v.device)
    valid = ordered - cumulative / indices > 0
    rho = int(torch.nonzero(valid)[-1])
    return (v - cumulative[rho] / (rho + 1)).clamp_min(0).to(values.dtype)


def paper_ranks(weights, rank):
    # Tiny roundoff at exact integer boundaries must not delete a rank.
    values = weights.double() * rank
    nearest = values.round()
    values = torch.where((values - nearest).abs() < 1e-6, nearest, values)
    result = values.floor().long()
    if result.sum() > round(float(weights.sum()) * rank):
        raise RuntimeError("Rank budget exceeded")
    return result


class SimplexController:
    """Single-adapter, equal-cost Linear/Linear8bitLt rank search.

    q_proj is the default: unlike q_proj+v_proj on GQA models it has equal
    per-rank parameter cost, so the paper's rank budget is a parameter budget.
    """
    def __init__(self, model, rank=8, interval=10, updates=3, weight_lr=0.1,
                 regularization=0.01):
        self.model = model
        self.layers = dict(
            (name, layer) for name, layer in model.named_modules()
            if isinstance(layer, LoraLayer) and "default" in layer.lora_A
        )
        if not self.layers or interval < 1 or updates < 1:
            raise ValueError("No adapters or invalid rank-search schedule")
        costs = {m.in_features + m.out_features for m in self.layers.values()}
        if len(costs) != 1:
            raise ValueError("Unequal adapter costs: use equal-shape targets (e.g. q_proj)")
        if any(m.r["default"] != rank or m.merged for m in self.layers.values()):
            raise ValueError("Expected unmerged uniform-rank default adapters")
        self.names = list(self.layers)
        self.indices = {n: i for i, n in enumerate(self.names)}
        self.weights = torch.nn.Parameter(torch.ones(
            len(self.names), device=next(iter(self.layers.values())).lora_A["default"].weight.device))
        self.rank, self.interval, self.updates = rank, interval, updates
        self.weight_lr, self.regularization = weight_lr, regularization
        self.completed = 0
        self.frozen = False
        self.handles = {}
        self.history = []
        self.budget = rank * len(self.names) * costs.pop()
        for name in self.names:
            self._hook(name)

    def _hook(self, name):
        i = self.indices[name]
        self.handles[name] = self.layers[name].lora_B["default"].register_forward_hook(
            lambda module, args, output, index=i: output * self.weights[index].to(output.dtype))

    def parameters(self):
        return [] if self.frozen else [self.weights]

    @torch.no_grad()
    def step(self, step):
        """Called after AdamW; returns True when A/B parameters were replaced."""
        if self.frozen:
            return False
        if self.weights.grad is None or not torch.isfinite(self.weights.grad).all():
            raise RuntimeError("Missing/nonfinite importance gradient")
        active = [self.indices[n] for n in self.layers]
        w = self.weights[active]
        w -= self.weight_lr * (self.weights.grad[active] + self.regularization * (w - 1))
        self.weights[active] = w
        self.weights.grad = None
        if step % self.interval:
            return False
        projected = project_simplex(self.weights[active], len(self.names))
        self.weights[active] = projected
        ranks = paper_ranks(projected, self.rank).tolist()
        self.completed += 1
        final = self.completed == self.updates
        rank_map = dict(zip(list(self.layers), ranks))
        for name, new_rank in rank_map.items():
            layer = self.layers[name]
            self.handles.pop(name).remove()
            if new_rank == 0:
                # Permanently retire a zero-rank adapter, matching upstream active set.
                parent_name, attr = name.rsplit(".", 1)
                setattr(self.model.get_submodule(parent_name), attr, layer.get_base_layer())
                del self.layers[name]
                self.weights[self.indices[name]] = 0
                continue
            # Standard LoRA initialization at every projection, as in paper.
            old_a = layer.lora_A["default"].weight
            a = torch.nn.Linear(layer.in_features, new_rank, bias=False,
                                device=old_a.device, dtype=old_a.dtype)
            b = torch.nn.Linear(new_rank, layer.out_features, bias=False,
                                device=old_a.device, dtype=old_a.dtype)
            torch.nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(b.weight)
            layer.lora_A["default"], layer.lora_B["default"] = a, b
            layer.r["default"] = new_rank
            layer.scaling["default"] = layer.lora_alpha["default"] / new_rank
            if not final:
                self._hook(name)
        if not self.layers:
            raise RuntimeError("Rank allocation retired every adapter")
        config = self.model.peft_config["default"]
        # PEFT wraps the original model in base_model.model; saved keys omit it.
        strip = lambda n: n.removeprefix("base_model.model.")
        config.target_modules = {strip(n) for n in self.layers}
        config.rank_pattern = {strip(n): m.r["default"] for n, m in self.layers.items()}
        count = sum(m.lora_A["default"].weight.numel() + m.lora_B["default"].weight.numel()
                    for m in self.layers.values())
        if count > self.budget:
            raise RuntimeError("Adapter parameter budget exceeded")
        self.history.append({"step": step, "ranks": rank_map,
                             "weights": projected.cpu().tolist(),
                             "adapter_parameters": count, "budget": self.budget})
        if final:
            self.weights.fill_(1)
            self.weights.requires_grad_(False)
            self.frozen = True
        return True


def rebuild_optimizer(model, previous, lr, weight_decay):
    """Drop replaced A/B moments, preserve surviving classifier state."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=weight_decay)
    if previous is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in previous.state:
                    optimizer.state[p] = previous.state[p]
    return optimizer
