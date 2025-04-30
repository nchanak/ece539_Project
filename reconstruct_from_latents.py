# reconstruct_from_latents.py
"""
Real compression round-trip and reconstruction script:
- Encodes a batch via your IV-VAE
- Compresses latents via Constriction
- Decompresses them back
- Reconstructs frames with VAE decoder
- Saves original vs. reconstructed images for comparison
"""
import os
import argparse
import importlib.machinery
import importlib.util

import torch
import numpy as np
import constriction
from torchvision.utils import save_image


def load_cfg(path):
    loader = importlib.machinery.SourceFileLoader("cfg", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to cfg.py containing get_model/get_loader")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    # set device
    device = torch.device(
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )

    # load model & data loader
    cfg = load_cfg(args.config)
    model = cfg.get_model().to(device).eval()
    loader = cfg.get_loader()

    # quantization range
    LOW, HIGH = -127, 128
    os.makedirs("recon_images", exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)  # (B, C, T, H, W)

        # 1) Encode
        if hasattr(model, "encode"):
            mu, logvar = model.encode(batch)
        else:
            out = model(batch)
            mu, logvar = (out[1], out[2]) if isinstance(out, tuple) else (out["mu"], out["logvar"])

        # 2) Quantize
        means = mu.detach().cpu().numpy().astype(np.float32)
        stds = (0.5 * logvar).exp().detach().cpu().numpy().astype(np.float32)
        ints = np.round(means).astype(np.int32).flatten()

        # 3) Compress
        qmodel = constriction.stream.model.QuantizedGaussian(LOW, HIGH)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(ints, qmodel, means.flatten(), stds.flatten())
        compressed = encoder.get_compressed()

        # 4) Decompress (updated signature)
        decoder = constriction.stream.queue.RangeDecoder(compressed)
        decoded_ints = decoder.decode(
            qmodel,
            means.flatten(),  # per-symbol means
            stds.flatten()    # per-symbol stddevs
        )
        decoded = np.array(decoded_ints, dtype=np.int32).reshape(means.shape)
        z = torch.from_numpy(decoded).view_as(mu).to(device).type_as(mu)

        # 5) Reconstruct
        if hasattr(model, "decode"):
            recon = model.decode(z)
        else:
            out = model(z)
            recon = out[0] if isinstance(out, tuple) else out["recon"]

        # 6) Save
        save_image(batch[0], f"recon_images/orig_{batch_idx}.png")
        save_image(recon[0], f"recon_images/recon_{batch_idx}.png")
        print(f"Batch {batch_idx} processed. Compressed bits: {compressed.size * compressed.dtype.itemsize * 8}")
        break  # remove to process all batches


if __name__ == "__main__":
    main()
