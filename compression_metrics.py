# compression_metrics.py
import argparse
import importlib.machinery
import importlib.util
import json
import time

import torch
import numpy as np
import constriction
import matplotlib.pyplot as plt  # required for plotting

def load_cfg(path):
    """Dynamically load a cfg.py file from an arbitrary path."""
    loader = importlib.machinery.SourceFileLoader("cfg", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod

def compute_kl(mu, logvar):
    """KL divergence of N(mu,σ²) vs N(0,1), summed over all dims (in nats)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to your cfg.py")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    # 1) load cfg.py
    cfg = load_cfg(args.config)

    # 2) build model & loader
    model = cfg.get_model()
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model.to(device).eval()
    loader = cfg.get_loader()

    total_kl_nats = 0.0
    total_bits_real = 0
    total_pixels = 0
    start_time = time.time()

    # We'll use a fixed integer range for all latents:
    LOW, HIGH = -127, 128

    for batch in loader:
        batch = batch.to(device)  # (B,C,T,H,W)

        # Encode
        if hasattr(model, "encode"):
            mu, logvar = model.encode(batch)
        else:
            out = model(batch)
            if isinstance(out, tuple) and len(out) >= 3:
                _, mu, logvar = out[:3]
            elif isinstance(out, dict):
                mu, logvar = out["mu"], out["logvar"]
            else:
                raise RuntimeError("Can't extract mu/logvar from model output")

        # Theoretical KL
        kl = compute_kl(mu, logvar)
        total_kl_nats += kl.item()

        # Quantize latents
        means = mu.detach().cpu().numpy().astype(np.float32)
        stds  = (0.5 * logvar).exp().detach().cpu().numpy().astype(np.float32)
        ints  = np.round(means).astype(np.int32).flatten()

        # Real compression via RangeEncoder
        qmodel  = constriction.stream.model.QuantizedGaussian(LOW, HIGH)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(ints, qmodel, means.flatten(), stds.flatten())
        compressed = encoder.get_compressed()

        # Count bits and pixels
        total_bits_real += compressed.size * compressed.dtype.itemsize * 8
        B, C, T, H, W = batch.shape
        total_pixels += B * T * H * W

    elapsed = time.time() - start_time

    # Convert metrics
    total_bytes_real = total_bits_real // 8
    bytes_per_pixel = total_bytes_real / total_pixels
    total_kl_bits = total_kl_nats / np.log(2)
    bpp_kl = total_kl_bits / total_pixels

    # Print and save histogram
    plt.hist(ints, bins=100)
    plt.title("Histogram of Quantized Latent Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("latent_histogram.png")

    # Output JSON with bytes
    result = {
        "config": args.config,
        "total_bytes": int(total_bytes_real),
        "bytes_per_pixel": float(bytes_per_pixel),
        "bpp_kl": float(bpp_kl),
        "frames": int(T),
        "seconds": float(elapsed)
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
