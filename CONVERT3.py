# CONVERT_rewrite.py
# Rewritten pipeline: GPU-batched dithering, same-line progress + timers,
# preserves original resolution/aspect math and ccframe output format.

import os
import sys
import math
import time
import gc
import json
import subprocess
import shutil
from PIL import Image
import cv2
import ffmpeg
import numpy as np
import torch
import multiprocessing as mp
import psutil
import re

# ---------------- CONFIG ----------------
BATCH_SIZE = 64             # adjustable: number of frames processed per GPU batch
OverideQueueSize = False
queueSize = 1
careAboutExtraEdgePixels = False   # preserved flag (behavior preserved in convert_image_to_ccframe)
profile = False
displayProgress = True
legacyGPUSupport = False
dither = False               # True => Floyd–Steinberg dithering; False => nearest color only
# ----------------------------------------

# Your palette (kept identical)
cc_palette = {
    1:    (255, 255, 255),  # white
    2:    (216, 127, 51),   # orange
    4:    (178, 76, 216),   # magenta
    8:    (102, 153, 216),  # lightBlue
    16:   (229, 229, 51),   # yellow
    32:   (127, 204, 25),   # lime
    64:   (242, 127, 165),  # pink
    128:  (76, 76, 76),     # gray
    256:  (153, 153, 153),  # lightGray
    512:  (76, 127, 153),   # cyan
    1024: (127, 63, 178),   # purple
    2048: (51, 76, 178),    # blue
    4096: (102, 76, 51),    # brown
    8192: (102, 127, 51),   # green
    16384:(153, 51, 51),    # red
    32768:(25, 25, 25),     # black
}

hex_map = {
    1:    "0", 2:    "1", 4:    "2", 8:    "3",
    16:   "4", 32:   "5", 64:   "6", 128:  "7",
    256:  "8", 512:  "9", 1024: "a", 2048: "b",
    4096: "c", 8192: "d", 16384:"e", 32768:"f",
}

# Precompute lookup table for fast string building
lookup = np.full(32769, ord("f"), dtype=np.uint8)
for k, v in hex_map.items():
    lookup[k] = ord(v)

# CUDA device and palette tensors (palette moved to device at runtime)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("\033[91mWARNING: CUDA not available. Running on CPU - this will be slow.\033[0m")

palette = torch.tensor(list(cc_palette.values()), dtype=torch.float32, device=device) / 255.0  # (P,3)
palette_keys = torch.tensor(list(cc_palette.keys()), device=device)  # (P,)

# ----------------- helpers -----------------
def get_fps(path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'json',
        path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    info = json.loads(result.stdout)
    r_frame_rate = info['streams'][0]['r_frame_rate']
    num, den = map(int, r_frame_rate.split('/'))
    return num / den

def get_video_resolution(path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    info = json.loads(result.stdout)
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    return width, height

# ---------------- dithering (batch) ----------------
@torch.no_grad()
def closest_cc_batch(img_batch):
    """
    img_batch: float tensor (B, H, W, 3) in [0,1] on device
    returns: tensor (B, H, W) of palette keys (int)
    """
    B, H, W, C = img_batch.shape
    flat = img_batch.reshape(-1, 3)  # (B*H*W, 3)
    dists = torch.cdist(flat, palette)  # (B*H*W, P)
    nearest = torch.argmin(dists, dim=1)  # (B*H*W,)
    nearest = nearest.view(B, H, W)
    keys = palette_keys[nearest]  # (B,H,W) of actual palette key values
    return keys

@torch.no_grad()
def closest_cc_dither_batch(img_batch):
    """
    Floyd–Steinberg dithering for a batch.
    img_batch: float tensor (B, H, W, 3) on device
    returns: tensor (B, H, W) of palette keys
    """
    B, H, W, C = img_batch.shape
    work = img_batch.clone()  # (B,H,W,3)

    out_idx = torch.empty((B, H, W), dtype=torch.int64, device=device)

    # iterate rows — vectorized over batch
    for y in range(H):
        row = work[:, y, :, :]  # (B, W, 3)
        flat_row = row.reshape(-1, 3)  # (B*W, 3)
        d = torch.cdist(flat_row, palette)  # (B*W, P)
        nearest_flat = torch.argmin(d, dim=1)  # (B*W,)
        nearest = nearest_flat.view(B, W)  # (B, W)
        out_idx[:, y, :] = palette_keys[nearest]  # store actual keys
        chosen = palette[nearest]  # (B, W, 3)  (indexing palette by nearest works)
        error = row - chosen  # (B, W, 3)

        if W > 1:
            work[:, y, 1:, :] += error[:, :-1, :] * (7.0 / 16.0)
        if y + 1 < H:
            if W > 1:
                work[:, y + 1, :-1, :] += error[:, 1:, :] * (3.0 / 16.0)
            work[:, y + 1, :, :] += error * (5.0 / 16.0)
            if W > 2:
                work[:, y + 1, 1:, :] += error[:, :-1, :] * (1.0 / 16.0)

        # clamp current and next row
        work[:, y, :, :].clamp_(0.0, 1.0)
        if y + 1 < H:
            work[:, y + 1, :, :].clamp_(0.0, 1.0)

    # at this point out_idx contains the palette keys chosen per row
    # out_idx is of dtype int64 and contains actual palette key values (like 1,2,4,...)
    return out_idx

# ---------------- write .ccframe ----------------
def write_ccframe_from_keys(keys_np, out_path):
    """
    keys_np: (H,W) numpy array of palette key integers
    Writes the .ccframe file at out_path in your expected format.
    """
    lua_lines = ["return {"]
    for row in keys_np:
        bg_line = bytes(lookup[row]).decode("ascii")
        text_line = " " * len(bg_line)
        text_color_line = "0" * len(bg_line)
        lua_lines.append('  {' + '"' + text_line + '", "' + text_color_line + '", "' + bg_line + '"},')
    lua_lines.append("}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lua_lines))

# ---------------- convert single image helper ----------------
def convert_image_to_ccframe(input_path="", img=None, output_path="", total_res=(-1, -1), aspect=(-1, -1), precalc=False):
    """
    Mirrors your original convert_image_to_ccframe behavior.
    If img provided (PIL.Image), uses it; otherwise loads from input_path.
    If not precalc, performs same calc/resizing logic you used (including 1.5 vertical factor).
    """
    ccname = ""
    if img is None:
        print("Loading image:", input_path)
        img = Image.open(input_path).convert("RGB")
        if output_path == "":
            parts = input_path.split(".")
            parts.pop()
            new_filename = ".".join(parts) + ".bmp"
            ccname = ".".join(parts) + ".ccframe"
            output_path = new_filename
    else:
        ccname = output_path

    if not precalc:
        width, height = img.size
        calc_width, calc_height = 0, 0
        target_aspect = aspect[0] / aspect[1]
        image_aspect = (width * 1.5) / height

        if image_aspect > target_aspect:
            calc_width = total_res[0]
            calc_height = total_res[0] / image_aspect
        else:
            calc_height = total_res[1]
            calc_width = total_res[1] * image_aspect

        calc = (math.floor(calc_width), math.floor(calc_height))

        # Respect original behavior: resize to calc (nearest)
        img = img.resize(calc, Image.Resampling.NEAREST)

    # convert to numpy and process via GPU (single-frame batch)
    np_img = np.array(img)  # H x W x 3 uint8
    if dither:
        tensor = torch.from_numpy(np_img[np.newaxis, ...]).to(device).float().div_(255.0)  # (1,H,W,3)
        keys = closest_cc_dither_batch(tensor)[0].cpu().numpy()  # (H,W)
    else:
        tensor = torch.from_numpy(np_img[np.newaxis, ...]).to(device).float().div_(255.0)
        keys = closest_cc_batch(tensor)[0].cpu().numpy()

    write_ccframe_from_keys(keys, ccname)
    return ccname

# ---------------- main mp4 -> frames (batch GPU) ----------------
def mp4_to_frames(input_path, targetFPS, total_res, aspect):
    start_total = time.time()

    output_dir = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper()
    print("Working Dir", output_dir)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Resolution calc (exactly as you wrote)
    width, height = get_video_resolution(input_path)
    calc_width, calc_height = 0, 0
    target_aspect = aspect[0] / aspect[1]
    image_aspect = (width * 1.5) / height

    if image_aspect > target_aspect:
        calc_width = total_res[0]
        calc_height = total_res[0] / image_aspect
    else:
        calc_height = total_res[1]
        calc_width = total_res[1] * image_aspect

    calc = (math.floor(calc_width), math.floor(calc_height))
    print("Calculated resolution to match aspect ratio:", calc)
    calc = (calc[0] // 2 * 2, calc[1] // 2 * 2)
    print("Rescaling to:", calc)

    # Resample FPS via ffmpeg (TMP.mp4)
    current_fps = get_fps(input_path)
    targetFPS = min(current_fps, targetFPS)
    print(f"Resampling {current_fps} to {targetFPS} FPS")

    if not legacyGPUSupport:
        (
            ffmpeg
            .input(input_path, hwaccel='cuda', hwaccel_device=0)
            .filter('fps', fps=targetFPS, round='up')
            .filter('scale', width=calc[0], height=calc[1])
            .output(output_dir + "/TMP.mp4", vcodec='h264_nvenc')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )
    else:
        (
            ffmpeg
            .input(input_path, hwaccel='cuda', hwaccel_device=0)
            .filter('fps', fps=targetFPS, round='up')
            .filter('scale', width=calc[0], height=calc[1])
            .output(output_dir + "/TMP.mp4", vcodec='libx264')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )

    tmp_path = os.path.join(output_dir, "TMP.mp4")
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        print("Error: Could not open resampled video (TMP.mp4).")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    maxCache = queueSize if OverideQueueSize else total_frames

    # Main loop: read frames, accumulate batches, send to GPU
    frame_idx = 0
    batch = []
    load_start = time.time()
    last_print_load = 0
    last_print_proc = 0
    proc_total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # convert BGR->RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(frame_rgb)

        # same-line loading progress
        if displayProgress and (frame_idx - last_print_load >= 1):
            print(f"Loading frame into memory {frame_idx + 1} / {total_frames}\033[K", end='\r', flush=True)
            last_print_load = frame_idx

        # when batch reaches BATCH_SIZE, process it
        if len(batch) >= BATCH_SIZE:
            proc_start = time.time()
            tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device).float().div_(255.0)  # (B,H,W,3)
            if dither:
                keys_batch = closest_cc_dither_batch(tensor)
            else:
                keys_batch = closest_cc_batch(tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            proc_elapsed = time.time() - proc_start
            proc_total_time += proc_elapsed

            # write out files for this batch
            base_idx = frame_idx - len(batch) + 1
            for i in range(len(batch)):
                keys_np = keys_batch[i].cpu().numpy()
                out_path = os.path.join(output_dir, f"frame_{base_idx + i:03}.ccframe")
                write_ccframe_from_keys(keys_np, out_path)

            # free memory & reset batch
            batch.clear()
            torch.cuda.empty_cache()
            gc.collect()

            # same-line processing progress
            if displayProgress:
                print(f"Processed up to frame {frame_idx + 1} / {total_frames} (batch) \033[K", end='\r', flush=True)
                last_print_proc = frame_idx

        frame_idx += 1

    # process any remaining frames in batch
    if len(batch) > 0:
        proc_start = time.time()
        tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device).float().div_(255.0)
        if dither:
            keys_batch = closest_cc_dither_batch(tensor)
        else:
            keys_batch = closest_cc_batch(tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        proc_elapsed = time.time() - proc_start
        proc_total_time += proc_elapsed

        base_idx = frame_idx - len(batch)
        for i in range(len(batch)):
            keys_np = keys_batch[i].cpu().numpy()
            out_path = os.path.join(output_dir, f"frame_{base_idx + i:03}.ccframe")
            write_ccframe_from_keys(keys_np, out_path)

        batch.clear()
        torch.cuda.empty_cache()
        gc.collect()

    cap.release()
    # remove TMP file
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    total_elapsed = time.time() - start_total
    print(f"\nDone. Total elapsed: {total_elapsed:.2f}s — GPU processing time (sum batches): {proc_total_time:.2f}s")
    print(f"Output folder: {output_dir}")

# ---------------- main ----------------
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    if len(sys.argv) < 7:
        print("Usage: python CONVERT_rewrite.py <media_file> <Block Width> <Block Height> <Cell # X> <Cell # Y> <Text Scale> <FPS <=20 (Optional)>")
        sys.exit(1)

    input_file = sys.argv[1]
    totalMonitors = (int(sys.argv[2]), int(sys.argv[3]))
    aspect = (int(sys.argv[4]), int(sys.argv[5]))
    scale = float(sys.argv[6])

    if len(sys.argv) > 7:
        if float(sys.argv[7]) > 20:
            print("\033[91mREQUESTED FPS TOO HIGH, SETTING TO 20\033[0m")
        targetFPS = min(float(sys.argv[7]), 20)
    else:
        targetFPS = 20

    # preserve your original total_res computation
    total_res = (
        math.floor((round(21.33266129032258 * (totalMonitors[0] / aspect[0]) - 6.645161290322656) * aspect[0]) / (scale * 2)),
        math.floor((round(14.222324046920821 * (totalMonitors[1] / aspect[1]) - 4.449596774193615) * aspect[1]) / (scale * 2))
    )
    print("In game resolution to match:", total_res)

    # dispatch
    if input_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        mp4_to_frames(input_file, targetFPS, total_res, aspect)
    elif input_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        # process single image (keeps original behavior)
        convert_image_to_ccframe(input_path=input_file, total_res=total_res, aspect=aspect, precalc=False)
    else:
        print("Unsupported file type.")
