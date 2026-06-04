"""
Efficiency analysis: measure parameters, FLOPs, inference time, energy estimation.
"""

import time
import torch
import torch.nn as nn
import numpy as np


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: nn.Module, input_size=(1, 3, 224, 224)) -> float:
    """Measure FLOPs using thop library."""
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.randn(input_size).to(device)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        flops = macs * 2  # MACs to FLOPs
        return flops
    except ImportError:
        print("[WARN] thop not installed, using torchprofile fallback")
        try:
            from torchprofile import profile_macs
            device = next(model.parameters()).device
            dummy = torch.randn(input_size).to(device)
            macs = profile_macs(model, (dummy,))
            return macs * 2
        except ImportError:
            print("[WARN] Neither thop nor torchprofile available, returning 0")
            return 0


def measure_inference_time(model: nn.Module, input_size=(1, 3, 224, 224),
                           warmup: int = 50, repeats: int = 1000) -> dict:
    """Measure inference time on GPU."""
    device = next(model.parameters()).device
    dummy = torch.randn(input_size).to(device)

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "fps": 1000.0 / float(np.mean(times)),
    }


def estimate_energy(inference_time_ms: float, tdp_watts: float = 350) -> float:
    """
    Estimate energy per inference in millijoules.
    RTX 3090 TDP ~350W. Use fraction based on utilization.
    """
    # Assume ~30% GPU utilization during single inference
    utilization = 0.30
    power_w = tdp_watts * utilization
    time_s = inference_time_ms / 1000.0
    energy_j = power_w * time_s
    return energy_j * 1000  # mJ


def efficiency_report(models_dict: dict, device: torch.device = None) -> dict:
    """Generate efficiency report for multiple models.

    Args:
        models_dict: {name: model_instance}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {}
    for name, model in models_dict.items():
        model = model.to(device)
        model.eval()

        params = count_parameters(model)
        flops = measure_flops(model)
        timing = measure_inference_time(model)
        energy = estimate_energy(timing["mean_ms"])

        report[name] = {
            "params_M": params / 1e6,
            "flops_G": flops / 1e9,
            "inference_ms": timing["mean_ms"],
            "inference_std_ms": timing["std_ms"],
            "fps": timing["fps"],
            "energy_mJ": energy,
        }

        print(f"{name}: {params/1e6:.1f}M params, {flops/1e9:.2f}G FLOPs, "
              f"{timing['mean_ms']:.2f}ms inference, {timing['fps']:.0f} fps, "
              f"{energy:.1f}mJ energy")

    return report
