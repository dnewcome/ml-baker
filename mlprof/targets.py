"""Catalog of common training-target instance types and their hardware specs.

The spec only declares ``instance_type``; the runner resolves the rest
(vCPU/RAM/GPU/VRAM/price/precision support) from this catalog. Accepts both
SageMaker-prefixed names (``ml.g5.xlarge``) and EC2 names (``g5.xlarge``) —
they normalize to the same entry.

Prices are USD/hr on-demand in ``us-east-1`` as of ``PRICE_AS_OF``. They are
*approximate* — for planning estimates only. A future iteration should fetch
live prices from the AWS Pricing API and pull per-region values.

Adding a new instance type: append to ``_CATALOG`` below. Adding a new GPU
model: append to ``_GPUS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PRICE_AS_OF = "2026-05-20"

# ---- GPU specs (model-level, instance-independent) ----

@dataclass(frozen=True)
class GpuSpec:
    model: str
    vram_gb: int
    supports_fp16: bool
    supports_bf16: bool          # Ampere+ (A10G, A100, H100); not on V100/T4


_GPUS: dict[str, GpuSpec] = {
    "T4":         GpuSpec("T4",         vram_gb=16, supports_fp16=True,  supports_bf16=False),
    "V100":       GpuSpec("V100",       vram_gb=16, supports_fp16=True,  supports_bf16=False),
    "V100-32GB":  GpuSpec("V100-32GB",  vram_gb=32, supports_fp16=True,  supports_bf16=False),
    "A10G":       GpuSpec("A10G",       vram_gb=24, supports_fp16=True,  supports_bf16=True),
    "A100-40GB":  GpuSpec("A100-40GB",  vram_gb=40, supports_fp16=True,  supports_bf16=True),
    "A100-80GB":  GpuSpec("A100-80GB",  vram_gb=80, supports_fp16=True,  supports_bf16=True),
    "H100":       GpuSpec("H100",       vram_gb=80, supports_fp16=True,  supports_bf16=True),
}


# ---- Instance specs ----

@dataclass(frozen=True)
class InstanceSpec:
    instance_type: str               # canonical (no "ml." prefix)
    vcpus: int
    ram_gb: int
    gpu_count: int                   # 0 for CPU-only instances
    gpu: GpuSpec | None              # None when gpu_count == 0
    on_demand_usd_per_hour: float
    family: Literal["cpu", "gpu"]
    region: str = "us-east-1"
    price_as_of: str = PRICE_AS_OF

    @property
    def total_vram_gb(self) -> int:
        return self.gpu_count * self.gpu.vram_gb if self.gpu else 0


def _gpu(model: str, count: int, vcpus: int, ram_gb: int,
         price: float, name: str) -> InstanceSpec:
    return InstanceSpec(
        instance_type=name, vcpus=vcpus, ram_gb=ram_gb,
        gpu_count=count, gpu=_GPUS[model],
        on_demand_usd_per_hour=price, family="gpu",
    )


def _cpu(vcpus: int, ram_gb: int, price: float, name: str) -> InstanceSpec:
    return InstanceSpec(
        instance_type=name, vcpus=vcpus, ram_gb=ram_gb,
        gpu_count=0, gpu=None,
        on_demand_usd_per_hour=price, family="cpu",
    )


# Approximate us-east-1 on-demand prices. Refresh from AWS Pricing API later.
_CATALOG: dict[str, InstanceSpec] = {
    # CPU — common for sklearn / spaCy training
    "c5.xlarge":    _cpu(vcpus=4,  ram_gb=8,   price=0.17, name="c5.xlarge"),
    "c5.4xlarge":   _cpu(vcpus=16, ram_gb=32,  price=0.68, name="c5.4xlarge"),
    "c5.9xlarge":   _cpu(vcpus=36, ram_gb=72,  price=1.53, name="c5.9xlarge"),
    "c6i.4xlarge":  _cpu(vcpus=16, ram_gb=32,  price=0.68, name="c6i.4xlarge"),
    "m5.4xlarge":   _cpu(vcpus=16, ram_gb=64,  price=0.77, name="m5.4xlarge"),

    # GPU — T4 (entry-level inference/training)
    "g4dn.xlarge":   _gpu("T4", 1, vcpus=4,   ram_gb=16,  price=0.526, name="g4dn.xlarge"),
    "g4dn.2xlarge":  _gpu("T4", 1, vcpus=8,   ram_gb=32,  price=0.752, name="g4dn.2xlarge"),
    "g4dn.12xlarge": _gpu("T4", 4, vcpus=48,  ram_gb=192, price=3.912, name="g4dn.12xlarge"),

    # GPU — A10G (mid-range; bf16 capable)
    "g5.xlarge":     _gpu("A10G", 1, vcpus=4,   ram_gb=16,  price=1.006, name="g5.xlarge"),
    "g5.2xlarge":    _gpu("A10G", 1, vcpus=8,   ram_gb=32,  price=1.212, name="g5.2xlarge"),
    "g5.12xlarge":   _gpu("A10G", 4, vcpus=48,  ram_gb=192, price=5.672, name="g5.12xlarge"),
    "g5.48xlarge":   _gpu("A10G", 8, vcpus=192, ram_gb=768, price=16.288, name="g5.48xlarge"),

    # GPU — V100 (older, no bf16)
    "p3.2xlarge":    _gpu("V100",      1, vcpus=8,  ram_gb=61,  price=3.06,  name="p3.2xlarge"),
    "p3.8xlarge":    _gpu("V100",      4, vcpus=32, ram_gb=244, price=12.24, name="p3.8xlarge"),
    "p3.16xlarge":   _gpu("V100",      8, vcpus=64, ram_gb=488, price=24.48, name="p3.16xlarge"),

    # GPU — A100 (high-end training)
    "p4d.24xlarge":  _gpu("A100-40GB", 8, vcpus=96, ram_gb=1152, price=32.77, name="p4d.24xlarge"),
    "p4de.24xlarge": _gpu("A100-80GB", 8, vcpus=96, ram_gb=1152, price=40.97, name="p4de.24xlarge"),

    # GPU — H100 (frontier)
    "p5.48xlarge":   _gpu("H100", 8, vcpus=192, ram_gb=2048, price=98.32, name="p5.48xlarge"),
}


def _normalize(instance_type: str) -> str:
    """Strip SageMaker ``ml.`` prefix so ``ml.g5.xlarge`` and ``g5.xlarge``
    resolve to the same entry."""
    return instance_type[3:] if instance_type.startswith("ml.") else instance_type


def resolve(instance_type: str) -> InstanceSpec:
    """Look up an instance by EC2 or SageMaker name. Raises if unknown."""
    key = _normalize(instance_type)
    if key not in _CATALOG:
        raise KeyError(
            f"Unknown instance type {instance_type!r}. "
            f"Known: {sorted(_CATALOG)} (or add it to mlprof.targets._CATALOG)."
        )
    return _CATALOG[key]


def known_instances() -> list[str]:
    """All instance types the catalog currently knows about, EC2-named."""
    return sorted(_CATALOG)


def register(spec: InstanceSpec) -> None:
    """Add or override an entry. For one-off custom instances or to inject
    fresher pricing without editing the source."""
    _CATALOG[_normalize(spec.instance_type)] = spec
