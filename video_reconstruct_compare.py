# video_reconstruct_compare.py
"""
Side-by-side comparison video of original vs. reconstructed frames,
using real Constriction compression on the VAE latents.
"""
import os, subprocess, cv2, numpy as np, torch, constriction
import importlib.machinery, importlib.util

def load_cfg(path):
    loader = importlib.machinery.SourceFileLoader("cfg", path)
    spec   = importlib.util.spec_from_loader(loader.name, loader)
    mod    = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod

def extract_video_frames(path, H, W):
    cap = cv2.VideoCapture(path); raw, norm = [], []
    while True:
        ret, frame = cap.read()
        if not ret: break
        f = cv2.resize(frame, (W, H))
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        raw.append(rgb); norm.append(rgb/255.0)
    cap.release()
    return np.array(raw), np.array(norm)

def create_sequences(frames, L):
    return np.array([frames[i:i+L] for i in range(len(frames)-L+1)])

def save_avi(orig, rec, path, fps=24):
    h, w = orig.shape[1], orig.shape[2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w*2, h))
    for o, r in zip(orig, rec):
        o_u = o.astype(np.uint8)
        r_u = (np.clip(r,0,1)*255).astype(np.uint8)
        combo = np.hstack((o_u, r_u))
        out.write(cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"[✓] Saved AVI → {path}")

def convert_avi_to_mp4(avi, mp4=None, crf=23, scale=4):
    if mp4 is None: mp4 = avi.replace('.avi', '.mp4')
    cmd = ['ffmpeg','-y','-i',avi,'-vf',f'scale=iw*{scale}:ih*{scale}','-vcodec','libx264','-crf',str(crf),mp4]
    subprocess.run(cmd, check=True)
    return mp4

def generate_video_comparison(video_name, cfg_path, seq_len, H, W, fps=24, scale=4):
    cfg   = load_cfg(cfg_path)
    model = cfg.get_model().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()

    vid_path = os.path.join("extracted_videos", video_name)
    raw, norm = extract_video_frames(vid_path, H, W)
    if len(norm) < seq_len:
        raise RuntimeError(f"Video too short ({len(norm)} frames) for seq_len={seq_len}")

    seqs = create_sequences(norm, seq_len)
    x = torch.from_numpy(seqs).permute(0,4,1,2,3).float().to(model.device)

    with torch.no_grad():
        mu, logvar = model.encode(x)
        means = mu.cpu().numpy().astype(np.float32)
        stds  = (0.5*logvar).exp().cpu().numpy().astype(np.float32)
        ints  = np.round(means).astype(np.int32).flatten()

        # compress
        LOW, HIGH = -127, 128
        qmodel = constriction.stream.model.QuantizedGaussian(LOW, HIGH)
        enc    = constriction.stream.queue.RangeEncoder()
        enc.encode(ints, qmodel, means.flatten(), stds.flatten())
        comp   = enc.get_compressed()

        # decompress **(corrected)**
        dec = constriction.stream.queue.RangeDecoder(comp)
        decoded_ints = dec.decode(qmodel,
                                  means.flatten(),
                                  stds.flatten())
        decoded = np.array(decoded_ints, dtype=np.int32).reshape(means.shape)
        z = torch.from_numpy(decoded).view_as(mu).to(model.device).float()

        recon = model.decode(z)
        recon_np = recon.cpu().permute(0,2,3,4,1).numpy()

    rec_frames = [recon_np[0,0]] + [recon_np[i,-1] for i in range(recon_np.shape[0])]
    orig_crop  = raw[:len(rec_frames)]

    avi = f"comp_{os.path.splitext(video_name)[0]}.avi"
    save_avi(orig_crop, rec_frames, avi, fps=fps)

    os.makedirs("comparisons", exist_ok=True)
    mp4 = convert_avi_to_mp4(avi, os.path.join("comparisons", avi.replace('.avi','.mp4')), scale=scale)
    os.remove(avi)
    print(f"Saved comparison → {mp4}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video_name", help="Filename in extracted_videos/")
    p.add_argument("cfg_path",   help="Path to cfg.py")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--h",       type=int, default=256)
    p.add_argument("--w",       type=int, default=256)
    args = p.parse_args()
    generate_video_comparison(args.video_name, args.cfg_path, args.seq_len, args.h, args.w)
