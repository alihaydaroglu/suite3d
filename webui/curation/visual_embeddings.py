from __future__ import annotations

import os
from typing import Literal, Optional

import numpy as np

from .composite_image import CompositeImageBuilder
from .vision_backbone import get_backbone


def run_visual_embedding_pipeline(
    stats_path: str,
    out_dir: str,
    backbone_name: Literal["dinov2", "resnet18", "resnet50"] = "dinov2",
    batch_size: int = 64,
    box_size: tuple = (5, 20, 20),
    device: Optional[str] = None,
    save_composites: bool = False,
    chunk_size: int = 2000,
) -> str:
    """
    End-to-end: stats → composite images → vision embeddings → .npy

    Processes in chunks to avoid holding all 224×224 composites in RAM.

    Returns:
        Path to saved embeddings file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load stats
    print(f"Loading stats from {stats_path}...")
    raw = np.load(stats_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()
    stats = list(raw)
    N = len(stats)
    print(f"Found {N} cells")

    # 2. Load backbone once
    print(f"Loading {backbone_name} backbone...")
    backbone = get_backbone(backbone_name, device=device)
    emb_dim = backbone.embedding_dim

    # 3. Build composite-image maker (builds KD-tree once)
    builder = CompositeImageBuilder(stats, box_size=box_size)

    # 4. Process in chunks: compose → extract → discard composites
    embeddings = np.zeros((N, emb_dim), dtype=np.float32)
    composites_to_save = [] if save_composites else None

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        indices = np.arange(start, end)
        print(f"Chunk [{start}:{end}] / {N}")

        composites = builder.build_all(indices, progress_every=500)
        embeddings[start:end] = backbone.extract(composites, batch_size=batch_size)

        if save_composites:
            composites_to_save.append(composites)

    # 5. Save embeddings
    emb_path = os.path.join(out_dir, f"visual_embeddings_{backbone_name}.npy")
    np.save(emb_path, embeddings)
    print(f"Saved visual embeddings to {emb_path}: shape {embeddings.shape}")

    if save_composites and composites_to_save:
        all_comp = np.concatenate(composites_to_save, axis=0)
        comp_path = os.path.join(out_dir, "composite_images.npy")
        np.save(comp_path, all_comp)
        print(f"Saved composite images to {comp_path}")

    return emb_path
