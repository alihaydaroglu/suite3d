"""Phase-resolved timing for the speed comparison.

Samples system-wide CPU% (per-core, summed) and used memory at a fixed
interval in a background thread, while a phase runs. Mirrors the
reviewer's reporting convention:

  avg_cpu_pct, peak_cpu_pct — %core-equivalent (1200% ≈ 12 cores busy)
  avg_mem_gb,  peak_mem_gb  — system used memory across the run

Usage
-----
    from timing_harness import PhaseTimer, write_row

    with PhaseTimer("suite3d", "registration", out_csv) as t:
        run_registration()

    # write_row is called automatically on __exit__
"""
from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "timing_harness needs psutil; install it in the container image."
    ) from e


SAMPLE_INTERVAL_S = 0.5


@dataclass
class PhaseRow:
    tool: str          # "suite3d" or "caiman"
    phase: str         # "registration" or "detection"
    instance: str      # AWS instance label, or "local" for smoke tests
    dataset: str
    subset: Optional[int]  # number of frames/volumes used, or None for full
    wall_s: float
    avg_cpu_pct: float
    peak_cpu_pct: float
    avg_mem_gb: float
    peak_mem_gb: float
    n_cpu_cores: int


class _SystemSampler(threading.Thread):
    """Polls system CPU% (per-core summed) and used memory at intervals."""

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        super().__init__(daemon=True)
        self._interval = interval_s
        self._stop_evt = threading.Event()
        self.cpu_samples: list[float] = []
        self.mem_samples_gb: list[float] = []

    def run(self) -> None:
        # Prime cpu_percent so first reading isn't 0.0.
        psutil.cpu_percent(interval=None, percpu=True)
        while not self._stop_evt.is_set():
            per_core = psutil.cpu_percent(interval=self._interval, percpu=True)
            self.cpu_samples.append(float(sum(per_core)))
            self.mem_samples_gb.append(psutil.virtual_memory().used / (1024 ** 3))

    def stop(self) -> None:
        self._stop_evt.set()


class PhaseTimer:
    def __init__(
        self,
        tool: str,
        phase: str,
        out_csv: os.PathLike,
        *,
        dataset: str,
        instance: str = "local",
        subset: Optional[int] = None,
    ):
        self.tool = tool
        self.phase = phase
        self.out_csv = Path(out_csv)
        self.dataset = dataset
        self.instance = instance
        self.subset = subset
        self._sampler: Optional[_SystemSampler] = None
        self._t0 = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._sampler = _SystemSampler()
        self._sampler.start()
        self._t0 = time.perf_counter()
        self.wall_s = 0.0
        print(f"[{self.tool}/{self.phase}] start", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        wall = time.perf_counter() - self._t0
        self.wall_s = wall
        assert self._sampler is not None
        self._sampler.stop()
        self._sampler.join(timeout=2.0)
        # Dump raw samples (always; useful even on crashes for postmortem)
        _dump_samples(
            self.out_csv, tool=self.tool, phase=self.phase,
            subset=self.subset,
            cpu_samples=self._sampler.cpu_samples,
            mem_samples_gb=self._sampler.mem_samples_gb,
            interval_s=SAMPLE_INTERVAL_S, t0=self._t0, wall_s=wall,
        )
        if exc_type is not None:
            # Phase crashed — don't pollute the CSV with a fake "completed"
            # row. The exception will still propagate up.
            print(
                f"[{self.tool}/{self.phase}] CRASHED after {wall:.1f}s "
                f"({exc_type.__name__})",
                flush=True,
            )
            return False
        cpu = self._sampler.cpu_samples or [0.0]
        mem = self._sampler.mem_samples_gb or [0.0]
        row = PhaseRow(
            tool=self.tool,
            phase=self.phase,
            instance=self.instance,
            dataset=self.dataset,
            subset=self.subset,
            wall_s=round(wall, 3),
            avg_cpu_pct=round(sum(cpu) / len(cpu), 1),
            peak_cpu_pct=round(max(cpu), 1),
            avg_mem_gb=round(sum(mem) / len(mem), 2),
            peak_mem_gb=round(max(mem), 2),
            n_cpu_cores=psutil.cpu_count(logical=True),
        )
        _append_row(self.out_csv, row)
        print(
            f"[{self.tool}/{self.phase}] done in {wall:.1f}s | "
            f"cpu avg {row.avg_cpu_pct:.0f}% peak {row.peak_cpu_pct:.0f}% | "
            f"mem avg {row.avg_mem_gb:.1f}G peak {row.peak_mem_gb:.1f}G",
            flush=True,
        )
        return False  # never suppress exceptions


def write_total_row(
    out_csv: os.PathLike,
    *,
    tool: str,
    dataset: str,
    instance: str,
    subset: Optional[int],
    walls_s: dict[str, float],
) -> None:
    """Append a phase=total row summing the wall times in walls_s.

    walls_s maps phase-label → seconds (only wall time is summed; the
    CPU/mem aggregates are left blank in the total row since they
    don't sum sensibly).
    """
    total = sum(walls_s.values())
    row = PhaseRow(
        tool=tool, phase="total", instance=instance, dataset=dataset,
        subset=subset, wall_s=round(total, 3),
        avg_cpu_pct=0.0, peak_cpu_pct=0.0,
        avg_mem_gb=0.0, peak_mem_gb=0.0,
        n_cpu_cores=psutil.cpu_count(logical=True),
    )
    _append_row(Path(out_csv), row)
    parts = " + ".join(f"{k}={v:.1f}s" for k, v in walls_s.items())
    print(f"[{tool}/total] {total:.1f}s  ({parts})", flush=True)


def _dump_samples(
    out_csv: Path, *, tool: str, phase: str, subset: Optional[int],
    cpu_samples, mem_samples_gb, interval_s: float, t0: float, wall_s: float,
) -> None:
    """Write raw per-sample arrays alongside the CSV.

    Filename: samples_<tool>_<phase>[_subset<N>].npz next to the CSV.
    Arrays: t_s (offset seconds from phase start), cpu_pct, mem_gb.
    """
    try:
        import numpy as np
    except ImportError:
        return  # silently skip; aggregates still written
    if not cpu_samples:
        return
    n = len(cpu_samples)
    # Sample i landed at t0 + (i+1)*interval_s. Use (i+1) so first sample is
    # not at t=0 (it followed a `cpu_percent(interval=interval_s)` call).
    t_s = np.arange(1, n + 1, dtype=np.float64) * float(interval_s)
    suf = "" if subset is None else f"_subset{subset}"
    fname = out_csv.parent / f"samples_{tool}_{phase}{suf}.npz"
    np.savez_compressed(
        fname,
        t_s=t_s,
        cpu_pct=np.asarray(cpu_samples, dtype=np.float32),
        mem_gb=np.asarray(mem_samples_gb, dtype=np.float32),
        wall_s=np.float64(wall_s),
    )


def _append_row(path: Path, row: PhaseRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(row).keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(asdict(row))
