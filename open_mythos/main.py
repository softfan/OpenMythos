from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# FPF registry: D1-D10 defects, evidence records, phase gates
# ---------------------------------------------------------------------------


class FPFStatus(str, Enum):
    """Adopt/Adapt/Reject status used by the FPF control plane."""

    PENDING = "PENDING"
    ADOPT = "ADOPT"
    ADAPT = "ADAPT"
    REJECT = "REJECT"


@dataclass
class DefectSpec:
    """
    Specification for one FPF-facing defect metric.

    direction:
        "max" means lower is better and violation occurs when value > threshold.
        "min" means higher is better and violation occurs when value < threshold.
    """

    defect_id: str
    name: str
    metric_name: str
    threshold: float
    direction: str = "max"
    critical: bool = False
    description: str = ""


DEFAULT_DEFECT_SPECS: Dict[str, DefectSpec] = {
    "D1": DefectSpec(
        "D1",
        "LTI Spectral Instability",
        "spectral_radius",
        0.95,
        "max",
        True,
        "Recurrent transition is non-contractive or close to unstable.",
    ),
    "D2": DefectSpec(
        "D2",
        "MoE Expert Collapse",
        "expert_entropy",
        0.20,
        "min",
        False,
        "Expert assignment entropy is too low; routing collapses.",
    ),
    "D3": DefectSpec(
        "D3",
        "ACT / Pondering Mis-Budgeting",
        "act_budget_error",
        0.25,
        "max",
        False,
        "Halting distribution is misaligned with compute/quality budget.",
    ),
    "D4": DefectSpec(
        "D4",
        "Predictive-Coding Error Drift",
        "pc_error",
        0.10,
        "max",
        False,
        "Prediction error fails to decrease or drifts upward.",
    ),
    "D5": DefectSpec(
        "D5",
        "Neuromod Runaway",
        "neuromod_magnitude",
        10.0,
        "max",
        True,
        "Neuromodulator or induced fast-weight norm grows unboundedly.",
    ),
    "D6": DefectSpec(
        "D6",
        "Metacognitive Over-Pruning",
        "overprune_score",
        0.20,
        "max",
        False,
        "Metacognitive gate prunes heavy reasoning too aggressively.",
    ),
    "D7": DefectSpec(
        "D7",
        "Debiasing Performance Collapse",
        "debias_loss_delta",
        0.05,
        "max",
        False,
        "Debiasing losses hurt main task quality beyond budget.",
    ),
    "D8": DefectSpec(
        "D8",
        "Fast-Weight Overfitting / Forgetting",
        "forgetting_delta",
        0.05,
        "max",
        False,
        "Sequence-local plasticity causes task/context forgetting.",
    ),
    "D9": DefectSpec(
        "D9",
        "Dual-Process Cross-Talk",
        "crosstalk_score",
        0.20,
        "max",
        False,
        "System-2 outputs leak into System-1 outside controlled interfaces.",
    ),
    "D10": DefectSpec(
        "D10",
        "Evidence-Graph Staleness",
        "evidence_age_days",
        30.0,
        "max",
        False,
        "Registry decisions rely on stale evidence.",
    ),
}


@dataclass
class EvidenceRecord:
    """Single timestamped metric observation written into the FPF evidence graph."""

    evidence_id: str
    pattern_name: str
    defect_id: str
    value: float
    timestamp: float
    claim_scope: str = ""
    polarity: str = "supportive"
    source: str = ""
    notes: str = ""


@dataclass
class PatternRecord:
    """
    FPF holonic pattern record.

    One record corresponds to one architectural pattern such as ParcaeLTI,
    HybridRecurrence, AMORGate, CoupledNeuromodMoR, or MoEFFN.
    """

    name: str
    impl: Optional[Any] = None
    tradition: str = "unspecified"
    claim_scope: str = "unspecified"
    polarity: str = "supportive"
    timespan: Tuple[Optional[float], Optional[float]] = (None, None)
    required_defects: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: FPFStatus = FPFStatus.PENDING
    decision_rationale: str = ""
    cl_penalty: float = 0.0
    evidence_graph_id: Optional[str] = None
    confidence_badge: Optional[float] = None
    kernel_impact: Dict[str, Any] = field(default_factory=dict)
    affected_core_principles: set[str] = field(default_factory=set)
    inline_table_active: bool = True
    g2_pack_id: Optional[str] = None
    autonomy_budget: Optional[float] = None
    prescribed_procedure: bool = False
    last_update: float = field(default_factory=time.time)


class FPFPatternRegistry:
    """
    Operational FPF control plane for OpenMythos.

    Responsibilities:
        1. Track architectural patterns.
        2. Store D1-D10 defect metrics.
        3. Emit Adopt/Adapt/Reject decisions.
        4. Maintain timestamped evidence records.
        5. Enforce phase gates P1->P2->P3->P4.
        6. Support G.2 migration packs, especially for D4 predictive-coding drift.
    """

    def __init__(
        self,
        defect_specs: Optional[Dict[str, DefectSpec]] = None,
        evidence_decay_lambda: float = 0.05,
    ) -> None:
        self.defect_specs = defect_specs or dict(DEFAULT_DEFECT_SPECS)
        self.patterns: Dict[str, PatternRecord] = {}
        self.evidence: List[EvidenceRecord] = []
        self.g2_packs: Dict[str, Dict[str, Any]] = {}
        self.evidence_decay_lambda = evidence_decay_lambda

    def register_pattern(
        self,
        name: str,
        impl: Optional[Any] = None,
        required_defects: Optional[Sequence[str]] = None,
        tradition: str = "unspecified",
        claim_scope: str = "unspecified",
        polarity: str = "supportive",
        autonomy_budget: Optional[float] = None,
        prescribed_procedure: bool = False,
        affected_core_principles: Optional[Iterable[str]] = None,
        g2_pack_id: Optional[str] = None,
    ) -> PatternRecord:
        """Register a holonic pattern with FPF metadata."""
        if name in self.patterns:
            rec = self.patterns[name]
            if impl is not None:
                rec.impl = impl
            return rec

        rec = PatternRecord(
            name=name,
            impl=impl,
            tradition=tradition,
            claim_scope=claim_scope,
            polarity=polarity,
            required_defects=list(required_defects or []),
            autonomy_budget=autonomy_budget,
            prescribed_procedure=prescribed_procedure,
            affected_core_principles=set(affected_core_principles or []),
            g2_pack_id=g2_pack_id,
        )
        self.patterns[name] = rec
        return rec

    def update_metrics(
        self,
        name: str,
        defect_metrics: Dict[str, float],
        source: str = "runtime",
        notes: str = "",
    ) -> None:
        """Update D1-D10 metrics and append timestamped evidence records."""
        if name not in self.patterns:
            self.register_pattern(name)

        rec = self.patterns[name]
        now = time.time()

        for defect_id, value in defect_metrics.items():
            if defect_id not in self.defect_specs:
                warnings.warn(f"Unknown defect id: {defect_id}")
                continue

            rec.metrics[defect_id] = float(value)
            self.evidence.append(
                EvidenceRecord(
                    evidence_id=f"{name}:{defect_id}:{int(now * 1000)}",
                    pattern_name=name,
                    defect_id=defect_id,
                    value=float(value),
                    timestamp=now,
                    claim_scope=rec.claim_scope,
                    polarity=rec.polarity,
                    source=source,
                    notes=notes,
                )
            )

        rec.last_update = now

    def decide(self, name: str) -> FPFStatus:
        """
        Adopt/Adapt/Reject decision.

        Decision order:
            1. Required metrics present?
            2. Kernel impact violation?
            3. Critical defect violation?
            4. RoC compliance warning?
            5. Non-critical defect violation?
            6. Otherwise ADOPT.
        """
        if name not in self.patterns:
            raise KeyError(f"Pattern {name!r} is not registered.")

        rec = self.patterns[name]

        missing = [d for d in rec.required_defects if d not in rec.metrics]
        if missing:
            rec.status = FPFStatus.PENDING
            rec.decision_rationale = f"Missing required defect metrics: {missing}"
            return rec.status

        # D10 is always computed from wall-clock evidence age.
        age_days = (time.time() - rec.last_update) / 86400.0
        rec.metrics["D10"] = age_days

        violations: List[Tuple[str, float, float, bool]] = []
        for defect_id, value in rec.metrics.items():
            spec = self.defect_specs.get(defect_id)
            if spec is None:
                continue

            if spec.direction == "max":
                violated = value > spec.threshold
            elif spec.direction == "min":
                violated = value < spec.threshold
            else:
                raise ValueError(f"Invalid defect direction: {spec.direction}")

            if violated:
                violations.append((defect_id, value, spec.threshold, spec.critical))

        kernel_violation = bool(
            rec.affected_core_principles.intersection({"A.5", "A.7"})
        )
        if kernel_violation:
            rec.status = FPFStatus.REJECT
            rec.decision_rationale = (
                f"Kernel-impact violation: {sorted(rec.affected_core_principles)}"
            )
            return rec.status

        critical = [v for v in violations if v[3]]
        if critical:
            rec.status = FPFStatus.REJECT
            rec.decision_rationale = f"Critical defects violated: {critical}"
            return rec.status

        roc = self._roc_compliance_check(rec)
        if roc["violates"]:
            rec.status = FPFStatus.ADAPT
            rec.cl_penalty += 0.10
            rec.decision_rationale = f"RoC compliance warning: {roc['warning']}"
            return rec.status

        if violations:
            rec.status = FPFStatus.ADAPT
            rec.cl_penalty += 0.05 * len(violations)
            rec.decision_rationale = f"Non-critical defects violated: {violations}"
        else:
            rec.status = FPFStatus.ADOPT
            rec.decision_rationale = "All required defect metrics within thresholds."

        return rec.status

    def _roc_compliance_check(self, rec: PatternRecord) -> Dict[str, Any]:
        """
        Check Rule-of-Causality compliance.

        FPF preference:
            - RoC: constraints, budgets, safety boundaries.
            - IoP: hard-coded procedure prescriptions.

        We warn when a pattern prescribes a procedure without declaring an
        autonomy budget.
        """
        is_iop = bool(rec.prescribed_procedure)
        has_budget = rec.autonomy_budget is not None
        violates = is_iop and not has_budget
        return {
            "is_iop": is_iop,
            "has_autonomy_budget": has_budget,
            "violates": violates,
            "warning": "IoP-style pattern without autonomy budget." if violates else "",
        }

    def check_phase_gate(self, target_phase: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        P1-P4 phase-gate state machine.

        target_phase:
            "P2" => permission to enter calibration/metacognition phase.
            "P3" => permission to enter predictive-coding/neuromod phase.
            "P4" => permission to enter bias-control/dual-process phase.
        """
        failures: List[str] = []

        def require_max(metric: str, threshold: float) -> None:
            value = metrics.get(metric)
            if value is None or value > threshold:
                failures.append(f"{metric}: {value} > {threshold}")

        def require_min(metric: str, threshold: float) -> None:
            value = metrics.get(metric)
            if value is None or value < threshold:
                failures.append(f"{metric}: {value} < {threshold}")

        if target_phase == "P2":
            require_max("D1", self.defect_specs["D1"].threshold)
            require_min("D2", self.defect_specs["D2"].threshold)
            require_max("D3", self.defect_specs["D3"].threshold)
        elif target_phase == "P3":
            require_max("D1", self.defect_specs["D1"].threshold)
            require_min("D2", self.defect_specs["D2"].threshold)
            require_max("D3", self.defect_specs["D3"].threshold)
            require_max("D6", self.defect_specs["D6"].threshold)
            if "coverage_error" in metrics:
                require_max("coverage_error", 0.05)
        elif target_phase == "P4":
            require_max("D4", self.defect_specs["D4"].threshold)
            require_max("D5", self.defect_specs["D5"].threshold)
            require_max("D8", self.defect_specs["D8"].threshold)
        else:
            raise ValueError(f"Unknown target phase: {target_phase}")

        return {
            "target_phase": target_phase,
            "pass": len(failures) == 0,
            "failures": failures,
        }

    def migrate_g2_for_d4(
        self,
        pack_id: str,
        traditions: List[Dict[str, Any]],
        operators: List[Dict[str, Any]],
        experiments: Optional[List[Dict[str, Any]]] = None,
        selected_variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        G.2 migration pack for D4 Predictive-Coding Error Drift.

        Operational meaning:
            - Gather competing predictive-coding traditions.
            - Encode TraditionCards and OperatorCards.
            - Evaluate variants on D4 metrics.
            - Mark one variant Adopt/Adapt/Reject using evidence.
        """
        pack = {
            "pack_id": pack_id,
            "defect": "D4",
            "purpose": "Predictive-Coding Error Drift mitigation",
            "tradition_cards": traditions,
            "operator_cards": operators,
            "experiments": experiments or [],
            "selected_variant": selected_variant,
            "created_at": time.time(),
        }
        self.g2_packs[pack_id] = pack

        for rec in self.patterns.values():
            if "D4" in rec.required_defects or rec.name.lower().startswith("predictive"):
                rec.g2_pack_id = pack_id
                rec.inline_table_active = False

        return pack

    def export_evidence_graph(self) -> Dict[str, Any]:
        """Export registry state as a JSON-serialisable evidence graph."""
        return {
            "patterns": {
                name: {
                    "status": rec.status.value,
                    "metrics": rec.metrics,
                    "tradition": rec.tradition,
                    "claim_scope": rec.claim_scope,
                    "polarity": rec.polarity,
                    "decision_rationale": rec.decision_rationale,
                    "cl_penalty": rec.cl_penalty,
                    "g2_pack_id": rec.g2_pack_id,
                    "inline_table_active": rec.inline_table_active,
                }
                for name, rec in self.patterns.items()
            },
            "evidence": [e.__dict__ for e in self.evidence],
            "g2_packs": self.g2_packs,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the evidence graph as JSON."""
        return json.dumps(self.export_evidence_graph(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    This extends the original Recurrent-Depth Transformer config with optional
    FPF-aligned control surfaces:

        - Parcae-style LTI stability and optional HybridRecurrence.
        - MoE routing metrics for D2 expert-collapse detection.
        - ACT/AMOR-style metacognitive depth control.
        - Predictive-coding residual modulation for D4.
        - CoupledNeuromodMoR for neuromodulated depth routing.
        - FPFPatternRegistry integration for D1-D10 defect tracking.

    The defaults preserve the original OpenMythos behavior:
        recurrence_type="parcae",
        predictive coding disabled,
        AMOR disabled,
        neuromodulated MoR disabled.
    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter

    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2

    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim

    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0

    # Stable recurrence
    recurrence_type: str = "parcae"  # "parcae" | "hybrid"
    input_injection_norm: bool = True
    hybrid_rank: int = 64
    hybrid_lambda_spec: float = 1e-4
    hybrid_correction_scale: float = 0.05

    # Predictive coding
    use_predictive_coding: bool = False
    pc_rank: int = 128
    pc_loss_weight: float = 0.01

    # AMOR-style metacognitive gate
    use_amor_gate: bool = False
    amor_entropy_dim: int = 32
    amor_target_gate_rate: float = 0.20
    amor_strength: float = 0.50

    # Coupled neuromodulation + MoR-style recursion routing
    use_coupled_neuromod_mor: bool = False
    neuromod_dim: int = 64
    mor_topk: int = 2
    mor_coupling_strength: float = 0.50
    mor_halt_inhibition: float = 0.50

    # SST-style loop state and anti-anchoring resets
    use_anti_anchoring: bool = False
    n_anchor_resets: int = 2
    use_sst_loop_state: bool = False
    sst_alpha: float = 0.90
    use_loop_index_embedding: bool = True

    # FPF/instrumentation
    enable_moe_metrics: bool = True
    enable_fpf_registry: bool = True
    rank_budget_enforced: bool = True
    fast_weight_rank: int = 0

    def validate(self) -> None:
        """Validate internal consistency of the configuration."""
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads.")

        if self.attn_type not in {"gqa", "mla"}:
            raise ValueError("attn_type must be 'gqa' or 'mla'.")

        if self.attn_type == "gqa":
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError("n_heads must be divisible by n_kv_heads for GQA.")
            if (self.dim // self.n_heads) % 2 != 0:
                raise ValueError("GQA head_dim must be even for RoPE.")

        if self.attn_type == "mla":
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even for MLA RoPE.")

        if self.n_experts_per_tok > self.n_experts:
            raise ValueError("n_experts_per_tok cannot exceed n_experts.")

        if self.recurrence_type not in {"parcae", "hybrid"}:
            raise ValueError("recurrence_type must be 'parcae' or 'hybrid'.")

        if self.rank_budget_enforced:
            total_rank = self.fast_weight_rank + self.lora_rank
            if total_rank > max(1, self.dim // 4):
                raise ValueError(
                    "RankBudgetConstraint violated: "
                    f"fast_weight_rank + lora_rank = {total_rank} > dim//4 = {self.dim // 4}"
                )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int,
    max_len: int,
    theta: float = 500000.0,
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Each position gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as a complex tensor so that rotation is a single pointwise multiply.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)

    Returns:
        complex64 tensor of shape (max_len, dim//2).
    """
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D complex number and
    multiplies by the precomputed phasor for that position, rotating the
    representation in the complex plane without changing its norm.

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- precomputed complex frequencies of shape (T, head_dim//2),
                     already sliced to exactly the positions being processed
                     (caller is responsible for correct start_pos offset)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2 (Dao et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    When flash-attn is installed, uses flash_attn_func which handles GQA natively
    (no KV head expansion needed) and is IO-bound-optimal. Inputs are cast to
    bfloat16 for flash_attn and restored to the original dtype afterward.
    Falls back to manual scaled dot-product attention when flash-attn is absent.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.dropout_p = cfg.dropout

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        if _HAS_FLASH_ATTN and q.is_cuda:
            # flash_attn_func expects (B, T, H, head_dim) — GQA is handled natively
            # (n_kv_heads < n_heads is supported without repeat_interleave).
            # causal=True when mask is present (full-sequence prefill/training);
            # causal=False for single-token decode where T=1 and mask is None.
            orig_dtype = q.dtype
            flash_dtype = torch.bfloat16
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                flash_dtype = torch.float16

            q_f = q.to(flash_dtype)
            k_f = k.to(flash_dtype)
            v_f = v.to(flash_dtype)

            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(
                q_f,
                k_f,
                v_f,
                dropout_p=dropout_p,
                causal=(mask is not None),
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # Fallback: manual scaled dot-product with explicit KV head expansion.
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)

            q = q.transpose(1, 2)   # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # Optimized
            attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5) 
            if mask is not None:
                attn = attn + mask

            attn = F.dropout(
                F.softmax(attn, dim=-1),
                p=self.dropout_p,
                training=self.training,
            )
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.

    Caches compressed KV latent c_kv plus RoPE keys, reconstructing K_nope and V
    on demand. This dramatically reduces KV cache memory compared with full K/V.
    """

    def __init__(self, cfg: MythosConfig):
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank,
            cfg.n_heads * cfg.qk_nope_head_dim,
            bias=False,
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank,
            cfg.n_heads * cfg.qk_rope_head_dim,
            bias=False,
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {
                "c_kv": c_kv.detach(),
                "k_rope": k_rope.detach(),
            }

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)

        k_nope = kv[..., : self.qk_nope_dim]
        v = kv[..., self.qk_nope_dim :]
        k = torch.cat([k_nope, k_rope], dim=-1)

        # attention
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.q_head_dim ** -0.5)
        if mask is not None:
            attn = attn + mask

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """Single SwiGLU feed-forward expert."""

    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _normalized_entropy_from_counts(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum().clamp(min=1)
    p = (counts.float() / total).clamp(min=1e-12)
    return -(p * p.log()).sum() / math.log(max(2, counts.numel()))


def _gini_from_counts(counts: torch.Tensor) -> torch.Tensor:
    x = counts.float().sort()[0]
    n = x.numel()
    if x.sum() <= 0:
        return torch.tensor(0.0, device=x.device)

    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return (2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.


    Adds runtime metrics:
        - expert_entropy
        - expert_gini
        - max_expert_fraction

    These feed D2 MoE Expert Collapse in FPFPatternRegistry.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.enable_metrics = cfg.enable_moe_metrics

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias adjusted externally during training; not a gradient param
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

        self.last_routing_stats: Dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        logits = self.router(flat)
        # Aux-loss-free load balancing (DeepSeek-V3): the bias shifts only the
        # selection of which experts fire so underused experts are picked more,
        # but the gating weights come from unbiased softmax scores so the bias
        # never shows up in the gradient.
        logits = self.router(flat)  # (B*T, n_experts), unbiased
        scores = F.softmax(logits, dim=-1)

        # DeepSeek-style aux-loss-free bias routing:
        # bias affects expert selection, not gradient-carrying gate weights.
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)

        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # routed expert dispatch (token-level scatter)
        out = torch.zeros_like(flat)

        for eid, expert in enumerate(self.routed_experts):
            tok_idx, rank_idx = torch.where(topk_idx == eid)
            if tok_idx.numel() == 0:
                continue
            out[tok_idx] += expert(flat[tok_idx]) * topk_scores[tok_idx, rank_idx, None]

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        if self.enable_metrics:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.flatten(), minlength=self.n_experts)
                entropy = _normalized_entropy_from_counts(counts)
                gini = _gini_from_counts(counts)
                self.last_routing_stats = {
                    "expert_entropy": float(entropy.detach().cpu()),
                    "expert_gini": float(gini.detach().cpu()),
                    "max_expert_fraction": float(
                        (counts.max().float() / counts.sum().clamp(min=1)).detach().cpu()
                    ),
                }

        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop-index embedding
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor,
    loop_t: int,
    loop_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape

    """
    loop_dim = min(loop_dim, h.shape[-1])
    loop_dim = loop_dim - (loop_dim % 2)
    if loop_dim <= 0:
        return h

    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]

    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb

    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: a shared low-rank down-projection and up-projection matrix B
    are shared across all loops, while a small per-loop scale vector shifts the
    effective transformation at each depth without adding significant parameters.

    delta(x, t) = (down(x) * scale[t]) @ B

    For depth extrapolation beyond max_loop_iters, the final learned scale is
    reused instead of indexing out of range.
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index used to look up the per-loop scale

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output
        """
        # Clamp for depth extrapolation: at inference n_loops can exceed the
        # training max_loop_iters. Iterations beyond the trained range reuse
        # the last learned per-loop scale rather than indexing out of range.
        max_t = self.scale.num_embeddings - 1
        t_idx = min(loop_t, max_t)
        s = self.scale(torch.tensor(t_idx, device=x.device))
        return (self.down(x) * s) @ self.B


# ---------------------------------------------------------------------------
# Predictive coding residual modulator
# ---------------------------------------------------------------------------


class PredictionErrorResidualModulator(nn.Module):
    """
    Thin predictive-coding wrapper for one recurrent residual path.

    It never modifies the Parcae/Hybrid A/B recurrence parameters directly.
    Instead, it predicts the residual and computes a channel-wise gamma in [0, 1]:

        residual_modulated = gamma(error, h, e) * residual

    This preserves the conflict-resolution rule:
        PC error may modulate residual compute, but must not destabilize A/B.
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.predictor = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, dim, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False),
            nn.Sigmoid(),
        )
        self.last_error: float = 0.0
        self.last_loss: torch.Tensor = torch.tensor(0.0)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.predictor(self.norm(h + e))
        error = residual - pred

        gamma = self.gate(torch.cat([h, e, error], dim=-1))
        residual_modulated = gamma * residual

        pc_loss = F.mse_loss(pred.float(), residual.detach().float())
        pc_error_scalar = error.pow(2).mean(dim=-1, keepdim=True)

        self.last_error = float(pc_error_scalar.mean().detach().cpu())
        self.last_loss = pc_loss

        return residual_modulated, pc_error_scalar.detach(), pc_loss


# ---------------------------------------------------------------------------
# AMOR-style metacognitive entropy/error gate
# ---------------------------------------------------------------------------


class AMORGate(nn.Module):
    """
    Entropy + prediction-error metacognitive gate.

    High entropy or high prediction error means "think more"; in RecurrentBlock
    the gate reduces ACT halting probability, thereby allocating more loops.
    """

    def __init__(
        self,
        dim: int,
        entropy_dim: int = 32,
        target_gate_rate: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Linear(dim, entropy_dim, bias=False)
        self.error_proj = nn.Linear(1, 1, bias=False)
        self.raw_tau = nn.Parameter(torch.tensor(0.0))
        self.log_alpha = nn.Parameter(torch.tensor(1.0))
        self.target_gate_rate = target_gate_rate

        self.last_gate_rate: float = 0.0
        self.last_entropy: float = 0.0
        self.last_reg_loss: torch.Tensor = torch.tensor(0.0)

    def forward(
        self,
        h: torch.Tensor,
        pc_error: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(h)
        p = F.softmax(logits.float(), dim=-1)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(max(2, logits.shape[-1]))

        if pc_error is None:
            err_term = torch.zeros_like(entropy)
        else:
            err_term = self.error_proj(torch.tanh(pc_error.float()))

        alpha = F.softplus(self.log_alpha) + 1e-4
        tau = torch.sigmoid(self.raw_tau)

        gate_prob = torch.sigmoid(alpha * (entropy - tau) + err_term)
        gate_rate = gate_prob.mean()
        reg_loss = (gate_rate - self.target_gate_rate) ** 2

        self.last_gate_rate = float(gate_rate.detach().cpu())
        self.last_entropy = float(entropy.mean().detach().cpu())
        self.last_reg_loss = reg_loss

        return gate_prob.to(h.dtype), reg_loss


# ---------------------------------------------------------------------------
# Coupled Neuromodulation + MoR-style depth routing
# ---------------------------------------------------------------------------


class CoupledNeuromodMoR(nn.Module):
    """
    Unified neuromodulated MoR-style depth router.

    Neuromodulation biases recursion probability but does not hard-force depth.
    This resolves the NeuromodRNN <-> MoR conflict by preserving stochastic
    routing while allowing salience/error signals to influence compute depth.
    """

    def __init__(
        self,
        hidden_dim: int,
        mod_dim: int = 64,
        topk: int = 2,
        coupling_strength: float = 0.5,
    ):
        super().__init__()
        self.topk = topk
        self.n_routes = max(2, topk * 2)
        self.coupling_strength = coupling_strength

        self.mod_net = nn.Sequential(
            nn.Linear(hidden_dim, mod_dim),
            nn.ReLU(),
            nn.Linear(mod_dim, 1),
            nn.Sigmoid(),
        )
        self.depth_router = nn.Linear(hidden_dim, 1)
        self.route_router = nn.Linear(hidden_dim, self.n_routes)

        self.last_metrics: Dict[str, float] = {}

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mod_scalar = self.mod_net(h)

        base_depth_logit = self.depth_router(h)
        depth_logit = base_depth_logit + self.coupling_strength * (2.0 * mod_scalar - 1.0)
        depth_gate = torch.sigmoid(depth_logit).squeeze(-1)

        route_logits = self.route_router(h) + self.coupling_strength * mod_scalar
        route_scores = F.softmax(route_logits, dim=-1)
        _, route_ids = route_scores.topk(min(self.topk, self.n_routes), dim=-1)

        with torch.no_grad():
            p = route_scores.clamp(min=1e-12)
            ent = -(p * p.log()).sum(dim=-1) / math.log(self.n_routes)
            self.last_metrics = {
                "neuromod_magnitude": float(mod_scalar.mean().detach().cpu()),
                "depth_gate_mean": float(depth_gate.mean().detach().cpu()),
                "route_entropy": float(ent.mean().detach().cpu()),
            }

        return depth_gate, route_ids, mod_scalar.squeeze(-1)


# ---------------------------------------------------------------------------
# SST-style loop state and anti-anchoring reset
# ---------------------------------------------------------------------------


class SSTLoopState(nn.Module):
    """
    SST-style recurrent loop state.

    In this implementation the SST state is local to the recurrent forward pass
    and can replace explicit loop-index embedding when enabled.
    """

    def __init__(self, dim: int, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Sigmoid())

    def forward(
        self,
        h: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.alpha * state + (1.0 - self.alpha) * h
        signal = self.proj(self.norm(state))
        signal = self.gate(h) * signal
        return signal, state


class AntiAnchoringReset(nn.Module):
    """
    Soft reset mechanism for anti-anchoring.

    Acts on h_t, not on recurrence matrices A/B, so it does not violate the
    Parcae spectral-stability invariant.
    """

    def __init__(self, dim: int, n_resets: int, max_loops: int):
        super().__init__()
        self.reset_points = [
            max_loops * (i + 1) // (n_resets + 1)
            for i in range(max(0, n_resets))
        ]
        self.reset_gate = nn.Sequential(nn.Linear(dim + 1, dim), nn.Sigmoid())
        self.re_anchor = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor, e: torch.Tensor, loop_t: int) -> torch.Tensor:
        if loop_t not in self.reset_points:
            return h

        denom = max(1, max(self.reset_points) if self.reset_points else 1)
        loop_signal = torch.full(
            (h.shape[0], h.shape[1], 1),
            float(loop_t) / float(denom),
            device=h.device,
            dtype=h.dtype,
        )
        reset_strength = self.reset_gate(torch.cat([h, loop_signal], dim=-1))
        h_fresh = self.re_anchor(e)
        return reset_strength * h_fresh + (1.0 - reset_strength) * h


class HypothesisBudgetAllocator:
    """
    Allocate loop budget across N anti-confirmation hypotheses.

    Uses calibration scores, with a minimum compute floor per hypothesis.
    """

    def __init__(self, n_hypotheses: int = 3, total_loops: int = 16, min_per: int = 2):
        self.n_hypotheses = n_hypotheses
        self.total_loops = total_loops
        self.min_per = min_per

    def allocate(self, calibration_scores: torch.Tensor) -> Dict[int, int]:
        if calibration_scores.numel() != self.n_hypotheses:
            raise ValueError("calibration_scores must have n_hypotheses elements.")

        min_total = self.min_per * self.n_hypotheses
        if self.total_loops < min_total:
            raise ValueError("total_loops too small for min_per budget.")

        probs = F.softmax(calibration_scores.float() / 0.5, dim=0)
        remaining = self.total_loops - min_total

        alloc = torch.full(
            (self.n_hypotheses,),
            self.min_per,
            dtype=torch.long,
            device=calibration_scores.device,
        )
        alloc = alloc + torch.floor(probs * remaining).long()

        diff = self.total_loops - int(alloc.sum().item())
        alloc[0] += diff

        return {i: int(alloc[i].item()) for i in range(self.n_hypotheses)}


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- precomputed RoPE frequencies
            mask      -- additive causal mask or None
            kv_cache  -- cache dict mutated in-place by the attention layer
            cache_key -- key identifying this layer in the cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction) and HybridRecurrence
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilize training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretization:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    where log_A and log_dt are learned parameters and exp ensures positivity.
    This makes looped model training robust to hyperparameter choices and stable
    even at high learning rates.

        h_{t+1} = A*h_t + B*LN(e) + residual

    A is diagonal and parameterized so that ρ(A) < 1 by construction:

        A_continuous = Diag(-exp(log_A))
        A_discrete   = exp(dt * A_continuous)

    This is the Parcae-style stability surface used by OpenMythos P1.
    """

    def __init__(self, dim: int, input_norm: bool = True):
        """
        Args:
            dim -- hidden state dimension; one scalar per channel for A and B
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))  # log of discretization step Δt
        self.B = nn.Parameter(torch.ones(dim) * 0.1)
        self.e_norm = RMSNorm(dim) if input_norm else nn.Identity()
        self.last_spec_loss: torch.Tensor = torch.tensor(0.0)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Clamp keeps the product finite in float32 for any gradient step size.
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def spectral_radius(self) -> float:
        with torch.no_grad():
            return float(self.get_A().abs().max().detach().cpu())

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A().to(device=h.device, dtype=h.dtype)
        B = self.B.to(device=h.device, dtype=h.dtype)
        e_inj = self.e_norm(e)
        self.last_spec_loss = torch.zeros((), device=h.device, dtype=h.dtype)
        return A * h + B * e_inj + transformer_out


class ParcaeLTI(LTIInjection):
    """Named alias/subclass for the Parcae-style stable LTI injection."""

    pass


class HybridRecurrence(LTIInjection):
    """
    Parcae-stable base plus low-rank bilinear correction.

    Base:
        h_lin = A*h + B*LN(e) + residual

    Correction:
        bilin = out_proj( x_proj(residual) * h_proj(h) )

    The correction is tanh-bounded and scaled, while a cheap norm proxy is
    returned via last_spec_loss for training regularization.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 64,
        lambda_spec: float = 1e-4,
        correction_scale: float = 0.05,
        input_norm: bool = True,
    ):
        super().__init__(dim, input_norm=input_norm)
        if rank <= 0:
            raise ValueError("HybridRecurrence rank must be positive.")

        self.rank = rank
        self.lambda_spec = lambda_spec
        self.correction_scale = correction_scale

        self.x_proj = nn.Linear(dim, rank, bias=False)
        self.h_proj = nn.Linear(dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, dim, bias=False)

    def _spectral_proxy(self) -> torch.Tensor:
        # Frobenius product is a cheap upper-bound-style proxy.
        wx = self.x_proj.weight.float().norm()
        wh = self.h_proj.weight.float().norm()
        wo = self.out_proj.weight.float().norm()
        return self.lambda_spec * wx * wh * wo

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        base = super().forward(h, e, transformer_out)

        x_proj = self.x_proj(transformer_out)
        h_proj = self.h_proj(h)
        bilin = self.out_proj(x_proj * h_proj)
        bilin = torch.tanh(bilin) * self.correction_scale

        self.last_spec_loss = self._spectral_proxy().to(device=h.device, dtype=h.dtype)
        return base + bilin


# ---------------------------------------------------------------------------
# ACT halting
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting mechanism."""

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    Core recurrent block: one TransformerBlock looped T times.

    Optional integrations:
        - HybridRecurrence.
        - PredictiveCodingResidualModulator.
        - AMORGate.
        - CoupledNeuromodMoR.
        - SST-style loop state.
        - AntiAnchoringReset.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)

        if cfg.recurrence_type == "hybrid":
            self.injection = HybridRecurrence(
                cfg.dim,
                rank=cfg.hybrid_rank,
                lambda_spec=cfg.hybrid_lambda_spec,
                correction_scale=cfg.hybrid_correction_scale,
                input_norm=cfg.input_injection_norm,
            )
        else:
            self.injection = LTIInjection(cfg.dim, input_norm=cfg.input_injection_norm)

        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)

        self.loop_dim = max(2, (cfg.dim // 8) - ((cfg.dim // 8) % 2))

        self.pc = (
            PredictionErrorResidualModulator(cfg.dim, cfg.pc_rank)
            if cfg.use_predictive_coding
            else None
        )
        self.amor = (
            AMORGate(
                cfg.dim,
                entropy_dim=cfg.amor_entropy_dim,
                target_gate_rate=cfg.amor_target_gate_rate,
            )
            if cfg.use_amor_gate
            else None
        )
        self.neuromod_mor = (
            CoupledNeuromodMoR(
                cfg.dim,
                mod_dim=cfg.neuromod_dim,
                topk=cfg.mor_topk,
                coupling_strength=cfg.mor_coupling_strength,
            )
            if cfg.use_coupled_neuromod_mor
            else None
        )
        self.anti_anchor = (
            AntiAnchoringReset(cfg.dim, cfg.n_anchor_resets, cfg.max_loop_iters)
            if cfg.use_anti_anchoring
            else None
        )
        self.sst = SSTLoopState(cfg.dim, alpha=cfg.sst_alpha) if cfg.use_sst_loop_state else None

        self.last_metrics: Dict[str, float] = {}
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        n_loops = int(n_loops or self.cfg.max_loop_iters)
        B, T, _ = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device, dtype=torch.float32)
        h_out = torch.zeros_like(h)
        loop_weights_sum = torch.zeros(B, T, device=h.device, dtype=torch.float32)

        aux_loss = torch.zeros((), device=h.device, dtype=h.dtype)
        pc_error_scalar: Optional[torch.Tensor] = None
        sst_state = torch.zeros_like(h) if self.sst is not None else None

        for t in range(n_loops):
            h_prev = h

            if self.sst is not None and sst_state is not None:
                sst_signal, sst_state = self.sst(h, sst_state)
                h_loop = h + sst_signal
            elif self.cfg.use_loop_index_embedding:
                h_loop = loop_index_embedding(h, t, self.loop_dim)
            else:
                h_loop = h

            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"

            residual = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            residual = residual + self.lora(residual, t)

            if self.pc is not None:
                residual, pc_error_scalar, pc_loss = self.pc(h_prev, e, residual)
                aux_loss = aux_loss + self.cfg.pc_loss_weight * pc_loss.to(aux_loss.dtype)

            h = self.injection(h_prev, e, residual)

            if self.anti_anchor is not None:
                h = self.anti_anchor(h, e, t)

            p = self.act(h)

            if self.amor is not None:
                gate_prob, amor_reg = self.amor(h, pc_error_scalar)
                p = p * (1.0 - self.cfg.amor_strength * gate_prob.squeeze(-1))
                aux_loss = aux_loss + amor_reg.to(aux_loss.dtype)

            if self.neuromod_mor is not None:
                depth_gate, _, _ = self.neuromod_mor(h)
                p = p * (1.0 - self.cfg.mor_halt_inhibition * depth_gate)

            p = p.clamp(1e-6, 1.0)

            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0.0, max=1.0)

            weight = torch.where(
                cumulative_p + p.float() >= self.cfg.act_threshold,
                remainder,
                p.float(),
            )
            weight = weight * still_running.float()

            h_out = h_out + weight.to(h.dtype).unsqueeze(-1) * h
            loop_weights_sum = loop_weights_sum + weight

            cumulative_p = cumulative_p + p.float() * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # With kv_cache, all loop depths must populate deterministic cache keys.
            if halted.all() and kv_cache is None:
                break

        still_running = ~halted
        if still_running.any():
            remainder = (1.0 - loop_weights_sum).clamp(min=0.0, max=1.0)
            final_weight = remainder * still_running.float()
            h_out = h_out + final_weight.to(h.dtype).unsqueeze(-1) * h
            loop_weights_sum = loop_weights_sum + final_weight

        spec_loss = getattr(self.injection, "last_spec_loss", None)
        if spec_loss is not None:
            aux_loss = aux_loss + spec_loss.to(device=h.device, dtype=aux_loss.dtype)

        self.last_aux_loss = aux_loss

        metrics = {
            "spectral_radius": self.injection.spectral_radius(),
            "aux_loss": float(aux_loss.detach().cpu()),
            "effective_weight_sum": float(loop_weights_sum.mean().detach().cpu()),
            "act_cumulative_mean": float(cumulative_p.mean().detach().cpu()),
            "act_budget_error": float(abs(loop_weights_sum.mean().detach().cpu().item() - 1.0)),
        }

        if isinstance(self.block.ffn, MoEFFN):
            metrics.update(self.block.ffn.last_routing_stats)

        if self.pc is not None:
            metrics["pc_error"] = float(self.pc.last_error)

        if self.amor is not None:
            metrics["amor_gate_rate"] = float(self.amor.last_gate_rate)
            metrics["amor_entropy"] = float(self.amor.last_entropy)
            metrics["overprune_score"] = max(
                0.0,
                self.cfg.amor_target_gate_rate - float(self.amor.last_gate_rate),
            )

        if self.neuromod_mor is not None:
            metrics.update(self.neuromod_mor.last_metrics)

        self.last_metrics = metrics
        return h_out


# ---------------------------------------------------------------------------
# Full OpenMythos model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """
    OpenMythos — Recurrent-Depth Transformer language model.

    Pipeline:
        Input tokens
             ↓
        Prelude
             ↓
        Recurrent Block
             ↓
        Coda
             ↓
        RMSNorm + tied LM head

    FPF extensions are optional and disabled by default unless set in config.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads,
            cfg.max_seq_len,
            cfg.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim,
            cfg.max_seq_len,
            cfg.rope_theta,
        )
        self.register_buffer("freqs_cis_mla", freqs_mla, persistent=False)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

        self.fpf_registry: Optional[FPFPatternRegistry] = (
            FPFPatternRegistry() if cfg.enable_fpf_registry else None
        )

        self._init_weights()
        self._register_default_fpf_patterns()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _register_default_fpf_patterns(self) -> None:
        """Register core OpenMythos patterns in the FPF registry."""
        if self.fpf_registry is None:
            return

        rec_name = "HybridRecurrence" if self.cfg.recurrence_type == "hybrid" else "ParcaeLTI"

        self.fpf_registry.register_pattern(
            rec_name,
            impl=self.recurrent.injection,
            required_defects=["D1"],
            tradition="stable-looped-transformers",
            claim_scope="recurrent-depth-stability",
            autonomy_budget=1.0,
        )
        self.fpf_registry.register_pattern(
            "MoEFFN",
            impl=self.recurrent.block.ffn,
            required_defects=["D2"],
            tradition="sparse-moe-routing",
            claim_scope="expert-diversity",
            autonomy_budget=1.0,
        )
        self.fpf_registry.register_pattern(
            "ACTHalting",
            impl=self.recurrent.act,
            required_defects=["D3"],
            tradition="adaptive-computation-time",
            claim_scope="depth-budgeting",
            autonomy_budget=1.0,
        )

        if self.cfg.use_predictive_coding:
            self.fpf_registry.register_pattern(
                "PredictiveCodingResidualModulator",
                impl=self.recurrent.pc,
                required_defects=["D4"],
                tradition="predictive-coding",
                claim_scope="pc-error-drift",
                autonomy_budget=1.0,
            )

        if self.cfg.use_amor_gate:
            self.fpf_registry.register_pattern(
                "AMORGate",
                impl=self.recurrent.amor,
                required_defects=["D6"],
                tradition="metacognitive-gating",
                claim_scope="reasoning-pruning",
                autonomy_budget=1.0,
            )

        if self.cfg.use_coupled_neuromod_mor:
            self.fpf_registry.register_pattern(
                "CoupledNeuromodMoR",
                impl=self.recurrent.neuromod_mor,
                required_defects=["D5"],
                tradition="neuromodulated-recursion",
                claim_scope="adaptive-depth-routing",
                autonomy_budget=1.0,
            )

    @staticmethod
    def _causal_mask(
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        start_pos: int = 0,
        kv_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build additive causal mask broadcastable over (B, H, T, S).

        Supports chunked decoding by allowing start_pos and kv_len.
        """
        S = kv_len if kv_len is not None else seq_len

        q_pos = torch.arange(start_pos, start_pos + seq_len, device=device)
        k_pos = torch.arange(S, device=device)

        mask = torch.zeros(seq_len, S, device=device, dtype=dtype)
        mask = mask.masked_fill(k_pos.unsqueeze(0) > q_pos.unsqueeze(1), float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through Prelude -> Recurrent Block -> Coda.

        If return_aux=True, returns (logits, aux_dict), where aux_dict contains
        auxiliary losses, runtime metrics, and D1-D10 FPF metrics.
        """
        T = input_ids.shape[1]
        device = input_ids.device

        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(
                f"Requested positions [{start_pos}, {start_pos + T}) exceed "
                f"max_seq_len={self.cfg.max_seq_len}."
            )

        x = self.embed(input_ids)

        freqs_buf = self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        freqs_cis = freqs_buf[start_pos : start_pos + T]

        # Preserve original decode behavior: for single-token decoding mask=None.
        if T > 1:
            kv_len = start_pos + T if kv_cache is not None else T
            mask = self._causal_mask(T, device, x.dtype, start_pos=start_pos, kv_len=kv_len)
        else:
            mask = None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        logits = self.head(self.norm(x))

        if not return_aux:
            return logits

        aux = {
            "aux_loss": self.recurrent.last_aux_loss,
            "metrics": self.recurrent.last_metrics,
            "fpf_metrics": self.collect_fpf_metrics(),
        }
        return logits, aux

    def collect_fpf_metrics(self) -> Dict[str, float]:
        """Map latest runtime metrics into D1-D10 defect metrics."""
        m = self.recurrent.last_metrics
        return {
            "D1": float(m.get("spectral_radius", 1.0)),
            "D2": float(m.get("expert_entropy", 0.0)),
            "D3": float(m.get("act_budget_error", 1.0)),
            "D4": float(m.get("pc_error", 0.0)),
            "D5": float(m.get("neuromod_magnitude", 0.0)),
            "D6": float(m.get("overprune_score", 0.0)),
            "D7": float(m.get("debias_loss_delta", 0.0)),
            "D8": float(m.get("forgetting_delta", 0.0)),
            "D9": float(m.get("crosstalk_score", 0.0)),
            "D10": 0.0,
        }

    def update_fpf_registry_from_last_run(self) -> Dict[str, str]:
        """Push latest metrics into FPFPatternRegistry and run decisions."""
        if self.fpf_registry is None:
            return {}

        d = self.collect_fpf_metrics()
        statuses: Dict[str, str] = {}

        rec_name = "HybridRecurrence" if self.cfg.recurrence_type == "hybrid" else "ParcaeLTI"
        self.fpf_registry.update_metrics(rec_name, {"D1": d["D1"]})
        statuses[rec_name] = self.fpf_registry.decide(rec_name).value

        self.fpf_registry.update_metrics("MoEFFN", {"D2": d["D2"]})
        statuses["MoEFFN"] = self.fpf_registry.decide("MoEFFN").value

        self.fpf_registry.update_metrics("ACTHalting", {"D3": d["D3"]})
        statuses["ACTHalting"] = self.fpf_registry.decide("ACTHalting").value

        if self.cfg.use_predictive_coding:
            self.fpf_registry.update_metrics(
                "PredictiveCodingResidualModulator",
                {"D4": d["D4"]},
            )
            statuses["PredictiveCodingResidualModulator"] = (
                self.fpf_registry.decide("PredictiveCodingResidualModulator").value
            )

        if self.cfg.use_amor_gate:
            self.fpf_registry.update_metrics("AMORGate", {"D6": d["D6"]})
            statuses["AMORGate"] = self.fpf_registry.decide("AMORGate").value

        if self.cfg.use_coupled_neuromod_mor:
            self.fpf_registry.update_metrics("CoupledNeuromodMoR", {"D5": d["D5"]})
            statuses["CoupledNeuromodMoR"] = (
                self.fpf_registry.decide("CoupledNeuromodMoR").value
            )

        return statuses

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive token generation with KV caching."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if max_new_tokens > self.cfg.max_output_tokens:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds "
                f"cfg.max_output_tokens={self.cfg.max_output_tokens}"
            )

        self.eval()
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]

        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1

            logits = self.forward(
                cur_ids,
                n_loops=n_loops,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                v, _ = logits.topk(k)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids
