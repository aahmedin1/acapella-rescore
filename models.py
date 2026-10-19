"""Load a Demucs-family separation model and expose it as a simple
``separate(window) -> vocals`` numpy function for the streaming engine.

Defaults to ``mdx_q`` (quantized, CPU-friendly). Any signature accepted by
``demucs_infer.pretrained.get_model`` works (htdemucs, htdemucs_ft, mdx, ...).
"""

from __future__ import annotations

import numpy as np
import torch

from demucs_infer.apply import apply_model
from demucs_infer.pretrained import get_model


def load_vocals_separator(model_name: str = "mdx_q",
                          device: str = "cpu",
                          num_threads: int | None = None,
                          single_submodel: bool = False):
    """Returns (separate_fn, info_dict).

    separate_fn(window: float32 (channels, n)) -> float32 (channels, n)
    containing the vocals-only estimate.
    """
    if num_threads:
        torch.set_num_threads(num_threads)
    torch.set_grad_enabled(False)

    model = get_model(model_name)
    if single_submodel and hasattr(model, "models"):
        # A BagOfModels averages several networks; taking one trades a
        # little quality for a large CPU speedup (needed for real-time).
        model = model.models[0]
    model.to(device)
    model.eval()

    sources = list(model.sources)
    try:
        vocals_idx = sources.index("vocals")
    except ValueError:
        raise RuntimeError(f"model {model_name} has no vocals stem: {sources}")
    sample_rate = int(model.samplerate)
    channels = int(model.audio_channels)

    def separate(window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32)
        if window.shape[0] != channels:
            raise ValueError(f"expected {channels} channels, got {window.shape}")
        mix = torch.from_numpy(window).unsqueeze(0).to(device)
        with torch.no_grad():
            out = apply_model(model, mix, shifts=0, split=False, device=device)
        vocals = out[0, vocals_idx].detach().cpu().numpy().astype(np.float32)
        return vocals

    info = {
        "name": model_name,
        "single_submodel": single_submodel,
        "sources": sources,
        "sample_rate": sample_rate,
        "channels": channels,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "device": device,
    }
    return separate, info


def load_stems_separator(model_name: str = "mdx_q",
                         stems: tuple = ("vocals", "drums"),
                         device: str = "cpu",
                         num_threads: int | None = None,
                         single_submodel: bool = False):
    """Like load_vocals_separator, but one model pass returns a dict of
    several stems: separate_fn(window) -> {stem_name: float32 (C, N)}.
    """
    if num_threads:
        torch.set_num_threads(num_threads)
    torch.set_grad_enabled(False)

    model = get_model(model_name)
    if single_submodel and hasattr(model, "models"):
        model = model.models[0]
    model.to(device)
    model.eval()

    sources = list(model.sources)
    idxs = {}
    for s in stems:
        try:
            idxs[s] = sources.index(s)
        except ValueError:
            raise RuntimeError(f"model {model_name} has no {s} stem: {sources}")
    sample_rate = int(model.samplerate)
    channels = int(model.audio_channels)

    def separate(window: np.ndarray) -> dict:
        window = np.asarray(window, dtype=np.float32)
        if window.shape[0] != channels:
            raise ValueError(f"expected {channels} channels, got {window.shape}")
        mix = torch.from_numpy(window).unsqueeze(0).to(device)
        with torch.no_grad():
            out = apply_model(model, mix, shifts=0, split=False, device=device)
        result = {}
        for name, i in idxs.items():
            result[name] = out[0, i].detach().cpu().numpy().astype(np.float32)
        return result

    info = {
        "name": model_name,
        "single_submodel": single_submodel,
        "sources": sources,
        "stems": list(stems),
        "sample_rate": sample_rate,
        "channels": channels,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "device": device,
    }
    return separate, info
