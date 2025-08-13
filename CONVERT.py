from PIL import Image
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import sys, os, math, time, cv2, ffmpeg, json, subprocess, numpy as np, re, shutil, gc, psutil, threading, cv2
import cProfile
from skimage.color import rgb2lab
from operator import itemgetter
import cupy as cp
from textwrap import dedent
from math import ceil
import numba as nb
from multiprocessing import Manager
import zlib, base64, textwrap


# Ensure environment variables for thread control before Pools are created
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

start = time.time()
frameCalcStart = 0

# # # PERFORMANCE SETTINGS
poolSize = 12
OverideQueueSize = False
queueSize = 1
careAboutExtraEdgePixels = True
profile = False
displayProgress = True
legacyGPUSupport = False
dither = False
HQDither = True

# Palette (official ComputerCraft default colors)
cc_palette = {
    1:     (240, 240, 240),  # White       #F0F0F0
    2:     (242, 178, 51),   # Orange      #F2B233
    4:     (229, 127, 216),  # Magenta     #E57FD8
    8:     (153, 178, 242),  # Light Blue  #99B2F2
    16:    (222, 222, 108),  # Yellow      #DEDE6C
    32:    (127, 204, 25),   # Lime        #7FCC19
    64:    (242, 178, 204),  # Pink        #F2B2CC
    128:   (76, 76, 76),     # Gray        #4C4C4C
    256:   (153, 153, 153),  # Light Gray  #999999
    512:   (76, 153, 178),   # Cyan        #4C99B2
    1024:  (178, 102, 229),  # Purple      #B266E5
    2048:  (51, 102, 204),   # Blue        #3366CC
    4096:  (127, 102, 76),   # Brown       #7F664C
    8192:  (87, 166, 78),    # Green       #57A64E
    16384: (204, 76, 76),    # Red         #CC4C4C
    32768: (17, 17, 17),     # Black       #111111
}
hex_map = {
    1:    "0",  # white
    2:    "1",  # orange
    4:    "2",  # magenta
    8:    "3",  # lightBlue
    16:   "4",  # yellow
    32:   "5",  # lime
    64:   "6",  # pink
    128:  "7",  # gray
    256:  "8",  # lightGray
    512:  "9",  # cyan
    1024: "a",  # purple
    2048: "b",  # blue
    4096: "c",  # brown
    8192: "d",  # green
    16384:"e",  # red
    32768:"f",  # black
}

# utility: pin workers to allowed cores (caller passes list of cores)
def init_worker(cores):
    p = psutil.Process()
    try:
        p.cpu_affinity(cores)
    except Exception:
        pass
    # reduce thread usage in each worker
    # try:
    #     torch.set_num_threads(1)
    # except Exception:
    #     pass
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

# ---------- sRGB -> Lab (CuPy) ----------
def rgb_to_lab_cp(rgb):
    rgb = cp.asarray(rgb, dtype=cp.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0

    # Flatten if needed
    orig_shape = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)

    # Linearize
    mask = rgb_flat <= 0.04045
    rgb_lin = cp.empty_like(rgb_flat)
    rgb_lin[mask] = rgb_flat[mask] / 12.92
    rgb_lin[~mask] = ((rgb_flat[~mask] + 0.055) / 1.055) ** 2.4

    # Matrix multiply
    M = cp.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=cp.float32)
    xyz = rgb_lin @ M.T

    # Normalize for D65 white point
    xyz /= cp.array([0.95047, 1.00000, 1.08883], dtype=cp.float32)

    # f(t) helper
    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    f = cp.empty_like(xyz)
    f[mask] = cp.cbrt(xyz[mask])
    f[~mask] = (kappa * xyz[~mask] + 16.0) / 116.0

    # Convert to Lab
    L = (116.0 * f[:, 1]) - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])

    lab = cp.stack([L, a, b], axis=1)
    return lab.reshape(orig_shape)

def rgb_to_lab_np(rgb):
    if rgb.max() > 1.0:
        rgb /= 255.0
    orig_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    mask = rgb <= 0.04045
    rgb_lin = np.empty_like(rgb)
    rgb_lin[mask] = rgb[mask] / 12.92
    rgb_lin[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4

    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = rgb_lin @ M.T
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    f = np.empty_like(xyz)
    f[mask] = np.cbrt(xyz[mask])
    f[~mask] = (kappa * xyz[~mask] + 16.0) / 116.0

    L = (116.0 * f[:, 1]) - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1).reshape(orig_shape)

# CPU versions of your color conversion (matches your GPU versions)
def srgb_to_linear_np(srgb):
    a = 0.055
    return np.where(srgb <= 0.04045,
                    srgb / 12.92,
                    ((srgb + a) / (1 + a)) ** 2.4)

def srgb_to_linear_cp(srgb):
    a = 0.055
    return cp.where(srgb <= 0.04045,
                    srgb / 12.92,
                    ((srgb + a) / (1 + a)) ** 2.4)




# Convert palette to LAB with same pipeline as image
palette_srgb = cp.array(list(cc_palette.values()), dtype=cp.float32) / 255.0
palette_lin = srgb_to_linear_cp(palette_srgb)
palette_lab = rgb_to_lab_cp(palette_lin)

# Store palette keys as CuPy array for mapping later
palette_keys = cp.array(list(cc_palette.values()), dtype=cp.uint8)  # (P,3)
palette_key_ids = cp.array(list(cc_palette.keys()), dtype=cp.int32)  # (P,)

#Raw image to cc color aproximation with no dithering
def closest_cc(img):
    """Map each pixel in img (H,W,3 uint8/float) to the nearest CC palette *key* (int).
       Returns HxW numpy array of CC keys (e.g. 1,2,4,8,...).
    """
    # Upload and normalize
    img_cp = cp.asarray(img, dtype=cp.float32)
    if img_cp.max() > 1.0:
        img_cp /= 255.0

    # Convert to LAB (we expect rgb_to_lab_cp to accept (H,W,3) and return (H,W,3))
    img_lab = rgb_to_lab_cp(img_cp)  # (H,W,3)

    H, W, _ = img_lab.shape
    flat = img_lab.reshape(-1, 3)  # (N,3)

    # Vectorized nearest neighbor search in Lab on GPU (squared distances)
    # p_norm + c_norm - 2 * (p @ c^T)
    p_norm = cp.sum(flat * flat, axis=1, keepdims=True)                 # (N,1)
    c_norm = cp.sum(palette_lab * palette_lab, axis=1).reshape(1, -1)    # (1,P)
    cross  = 2.0 * (flat @ palette_lab.T)                               # (N,P)
    dists  = p_norm + c_norm - cross                                    # (N,P)
    best_idx = cp.argmin(dists, axis=1)                                 # (N,)

    # Map palette index -> ComputerCraft key id and return on CPU
    keys_gpu = palette_key_ids[best_idx].reshape(H, W)                   # (H,W) on GPU
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(keys_gpu)

# ---------- Tile kernel (works for either horizontal or vertical) ----------
# Kernel: one block per tile; each block processes its tile serially (prevents in-block races).
# Host will launch tiles on each diagonal; tiles on same diagonal are independent.
cuda_src = dedent(r'''
extern "C" {
__device__ inline float sqr(float x){ return x*x; }

__global__ void tile_pass(
    float* img,          // Lab image flattened: (H*W*3) contiguous
    int H, int W,
    const float* palette, // P x 3
    int P,
    int tile_size,
    const int* tile_tx,  // arrays of tile coords for this launch
    const int* tile_ty,
    int num_tiles,
    int pass_mode,       // 0 = horizontal (row-major), 1 = vertical (col-major)
    int* out_idx         // H*W ints (row-major)
){
    int t = blockIdx.x; 
    if (t >= num_tiles) return;

    int tx = tile_tx[t];
    int ty = tile_ty[t];

    int x0 = tx * tile_size;
    int y0 = ty * tile_size;
    int x1 = x0 + tile_size; if (x1 > W) x1 = W;
    int y1 = y0 + tile_size; if (y1 > H) y1 = H;

    // process in serial order inside tile depending on pass_mode
    if (pass_mode == 0) {
        // horizontal: y outer, x inner (row-major)
        for (int y = y0; y < y1; ++y){
            for (int x = x0; x < x1; ++x){
                int base = (y * W + x) * 3;
                float L = img[base + 0];
                float A = img[base + 1];
                float B = img[base + 2];

                // nearest palette brute force
                int best = 0;
                float bestd = 1e30f;
                for (int pi = 0; pi < P; ++pi){
                    float pd0 = palette[pi*3 + 0];
                    float pd1 = palette[pi*3 + 1];
                    float pd2 = palette[pi*3 + 2];
                    float d = sqr(L - pd0) + sqr(A - pd1) + sqr(B - pd2);
                    if (d < bestd){ bestd = d; best = pi; }
                }
                out_idx[y * W + x] = best;
                float c0 = palette[best*3 + 0];
                float c1 = palette[best*3 + 1];
                float c2 = palette[best*3 + 2];
                float e0 = L - c0;
                float e1 = A - c1;
                float e2 = B - c2;

                // Floyd-Steinberg weights (same as before)
                // x+1, y   : 7/16
                if (x + 1 < W){
                    int b1 = (y * W + (x+1)) * 3;
                    img[b1 + 0] += e0 * (7.0f/16.0f);
                    img[b1 + 1] += e1 * (7.0f/16.0f);
                    img[b1 + 2] += e2 * (7.0f/16.0f);
                }
                // x-1, y+1 : 3/16
                if (y + 1 < H && x - 1 >= 0){
                    int b2 = ((y+1) * W + (x-1)) * 3;
                    img[b2 + 0] += e0 * (3.0f/16.0f);
                    img[b2 + 1] += e1 * (3.0f/16.0f);
                    img[b2 + 2] += e2 * (3.0f/16.0f);
                }
                // x, y+1 : 5/16
                if (y + 1 < H){
                    int b3 = ((y+1) * W + x) * 3;
                    img[b3 + 0] += e0 * (5.0f/16.0f);
                    img[b3 + 1] += e1 * (5.0f/16.0f);
                    img[b3 + 2] += e2 * (5.0f/16.0f);
                }
                // x+1, y+1 : 1/16
                if (y + 1 < H && x + 1 < W){
                    int b4 = ((y+1) * W + (x+1)) * 3;
                    img[b4 + 0] += e0 * (1.0f/16.0f);
                    img[b4 + 1] += e1 * (1.0f/16.0f);
                    img[b4 + 2] += e2 * (1.0f/16.0f);
                }
            }
        }
    } else {
        // vertical: x outer, y inner (column-major)
        for (int x = x0; x < x1; ++x){
            for (int y = y0; y < y1; ++y){
                int base = (y * W + x) * 3;
                float L = img[base + 0];
                float A = img[base + 1];
                float B = img[base + 2];

                int best = 0;
                float bestd = 1e30f;
                for (int pi = 0; pi < P; ++pi){
                    float pd0 = palette[pi*3 + 0];
                    float pd1 = palette[pi*3 + 1];
                    float pd2 = palette[pi*3 + 2];
                    float d = sqr(L - pd0) + sqr(A - pd1) + sqr(B - pd2);
                    if (d < bestd){ bestd = d; best = pi; }
                }
                out_idx[y * W + x] = best;
                float c0 = palette[best*3 + 0];
                float c1 = palette[best*3 + 1];
                float c2 = palette[best*3 + 2];
                float e0 = L - c0;
                float e1 = A - c1;
                float e2 = B - c2;

                // vertical-adapted diffusion (same as your vertical pass)
                if (y + 1 < H){
                    int b1 = ((y+1) * W + x) * 3;
                    img[b1 + 0] += e0 * (7.0f/16.0f);
                    img[b1 + 1] += e1 * (7.0f/16.0f);
                    img[b1 + 2] += e2 * (7.0f/16.0f);
                }
                if (x + 1 < W && y - 1 >= 0){
                    int b2 = ((y-1) * W + (x+1)) * 3;
                    img[b2 + 0] += e0 * (3.0f/16.0f);
                    img[b2 + 1] += e1 * (3.0f/16.0f);
                    img[b2 + 2] += e2 * (3.0f/16.0f);
                }
                if (x + 1 < W){
                    int b3 = (y * W + (x+1)) * 3;
                    img[b3 + 0] += e0 * (5.0f/16.0f);
                    img[b3 + 1] += e1 * (5.0f/16.0f);
                    img[b3 + 2] += e2 * (5.0f/16.0f);
                }
                if (x + 1 < W && y + 1 < H){
                    int b4 = ((y+1) * W + (x+1)) * 3;
                    img[b4 + 0] += e0 * (1.0f/16.0f);
                    img[b4 + 1] += e1 * (1.0f/16.0f);
                    img[b4 + 2] += e2 * (1.0f/16.0f);
                }
            }
        }
    }
}
} // extern C
''')

_tile_kernel = cp.RawKernel(cuda_src, 'tile_pass')

# ---------- Host wrapper ----------
def closest_cc_dither(image_array, tile_size=32, use_fp16=False):
    """
    Drop-in replacement: image_array is HxWx3 uint8 (0..255) or float32 0..1 numpy array.
    cc_palette: dict mapping key->(r,g,b) ints 0..255
    Returns: HxW numpy array of palette keys (ints)
    """
    # prepare palette Lab on GPU
    pal_vals = np.array(list(cc_palette.values()), dtype=np.float32) / 255.0  # (P,3)
    pal_gpu_srgb = cp.array(pal_vals, dtype=cp.float32)
    pal_gpu_lab = rgb_to_lab_cp(pal_gpu_srgb[cp.newaxis, :, :]).reshape(-1, 3)  # (P,3)

    # prepare image Lab on GPU
    if image_array.dtype == np.uint8:
        img_gpu = cp.array(image_array, dtype=cp.float32) / 255.0
    else:
        img_gpu = cp.array(image_array, dtype=cp.float32)
    img_lab = rgb_to_lab_cp(img_gpu)  # (H,W,3)

    if use_fp16:
        img_lab = img_lab.astype(cp.float16)
        pal_gpu_lab = pal_gpu_lab.astype(cp.float16)

    H, W, _ = img_lab.shape
    P = pal_gpu_lab.shape[0]

    # flattened images and index buffers (Lab storage will be updated in-place)
    img_flat = cp.ascontiguousarray(img_lab.ravel())
    idx_h = cp.empty((H * W,), dtype=cp.int32)
    idx_v = cp.empty((H * W,), dtype=cp.int32)

    tiles_x = ceil(W / tile_size)
    tiles_y = ceil(H / tile_size)

    # Helper to process pass with wavefront tiling
    def run_pass(pass_mode, out_idx_buffer, img_source_flat):
        # we'll mutate the provided img_source_flat in-place
        # iterate diagonals
        maxd = tiles_x + tiles_y - 2
        for d in range(0, maxd + 1):
            # collect tiles with tx + ty == d
            tx_list = []
            ty_list = []
            for ty in range(max(0, d - (tiles_x - 1)), min(tiles_y - 1, d) + 1):
                tx = d - ty
                if tx < 0 or tx >= tiles_x:
                    continue
                tx_list.append(tx)
                ty_list.append(ty)
            if not tx_list:
                continue
            tx_gpu = cp.array(tx_list, dtype=cp.int32)
            ty_gpu = cp.array(ty_list, dtype=cp.int32)
            num_tiles = tx_gpu.size

            # launch kernel with one block per tile; blockDim=1 serially executes tile in device code
            _tile_kernel(
                (int(num_tiles),), (1,),
                (img_source_flat, np.int32(H), np.int32(W),
                 pal_gpu_lab.astype(cp.float32).ravel(), np.int32(P),
                 np.int32(tile_size),
                 tx_gpu, ty_gpu, np.int32(num_tiles),
                 np.int32(pass_mode),
                 out_idx_buffer)
            )
            # GPU kernel done for this diagonal; next diagonal safe to proceed
        # end for diagonals

    # Horizontal pass: row-major scanning within tiles
    run_pass(pass_mode=0, out_idx_buffer=idx_h, img_source_flat=img_flat)

    # Recreate original Lab image for vertical pass (to mimic independent passes)
    img_lab2 = rgb_to_lab_cp(img_gpu) if not use_fp16 else rgb_to_lab_cp(img_gpu).astype(cp.float16)
    img_flat2 = cp.ascontiguousarray(img_lab2.ravel())

    # Vertical pass: column-major scanning within tiles
    run_pass(pass_mode=1, out_idx_buffer=idx_v, img_source_flat=img_flat2)

    # reshape index buffers
    idx_h = idx_h.reshape((H, W))
    idx_v = idx_v.reshape((H, W))

    # map indices to Lab palette and average
    pal_lab = pal_gpu_lab  # (P,3)
    lab_h = pal_lab[idx_h.ravel()].reshape((H, W, 3))
    lab_v = pal_lab[idx_v.ravel()].reshape((H, W, 3))
    lab_avg = (lab_h + lab_v) * 0.5

    # remap averaged Lab to nearest palette (vectorized)
    flat_avg = lab_avg.reshape(-1, 3)
    p_norm = cp.sum(flat_avg * flat_avg, axis=1, keepdims=True)        # (N,1)
    c_norm = cp.sum(pal_lab * pal_lab, axis=1).reshape(1, -1)           # (1,P)
    cross = 2.0 * (flat_avg @ pal_lab.T)
    dists = p_norm + c_norm - cross
    final_idx_flat = cp.argmin(dists, axis=1).reshape((H, W))

    # map to palette keys and return CPU numpy
    palette_keys = np.array(list(cc_palette.keys()), dtype=np.int32)
    keys_gpu = cp.array(palette_keys)[final_idx_flat]
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(keys_gpu)

def closest_cc_dither_hq(image_array, use_fp16=False):

    """
    High-quality Floyd–Steinberg dithering using CuPy.
    Processes the image strictly pixel-by-pixel (left-to-right, top-to-bottom)
    with error diffusion.
    """

    # Prepare palette Lab on GPU
    pal_vals = np.array(list(cc_palette.values()), dtype=np.float32) / 255.0
    pal_gpu_srgb = cp.array(pal_vals, dtype=cp.float32)
    pal_gpu_lab = rgb_to_lab_cp(pal_gpu_srgb[cp.newaxis, :, :]).reshape(-1, 3)  # (P,3)

    # Prepare image Lab on GPU
    if image_array.dtype == np.uint8:
        img_gpu = cp.array(image_array, dtype=cp.float32) / 255.0
    else:
        img_gpu = cp.array(image_array, dtype=cp.float32)
    img_lab = rgb_to_lab_cp(img_gpu)  # (H,W,3)

    if use_fp16:
        img_lab = img_lab.astype(cp.float16)
        pal_gpu_lab = pal_gpu_lab.astype(cp.float16)

    H, W, _ = img_lab.shape
    out_idx = cp.empty((H, W), dtype=cp.int32)

    # Process pixels in scanline order
    for y in range(H):
        for x in range(W):
            pix = img_lab[y, x]
            # Find nearest palette color
            diffs = pal_gpu_lab - pix
            dists = cp.sum(diffs * diffs, axis=1)
            best_idx = int(cp.argmin(dists))
            out_idx[y, x] = best_idx

            chosen = pal_gpu_lab[best_idx]
            err = pix - chosen

            # Floyd–Steinberg diffusion
            if x + 1 < W:
                img_lab[y, x+1] += err * (7.0/16.0)
            if y + 1 < H:
                if x > 0:
                    img_lab[y+1, x-1] += err * (3.0/16.0)
                img_lab[y+1, x] += err * (5.0/16.0)
                if x + 1 < W:
                    img_lab[y+1, x+1] += err * (1.0/16.0)
        print(f"Processing pixels {x+y*W} / {H*W}", end='\r', flush=True)


    # Map palette indices to CC keys
    palette_keys = np.array(list(cc_palette.keys()), dtype=np.int32)
    keys_gpu = cp.array(palette_keys)[out_idx]
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(keys_gpu)

@nb.njit(cache=True, fastmath=True)
def _fs_dither_lab(img_lab, palette_lab, palette_keys):
    H, W, _ = img_lab.shape
    P = palette_lab.shape[0]
    out_keys = np.empty((H, W), dtype=np.int32)

    for y in range(H):
        for x in range(W):
            # Find nearest palette color in Lab space
            L, A, B = img_lab[y, x]
            best_idx = 0
            best_dist = 1e30
            for pi in range(P):
                dL = L - palette_lab[pi, 0]
                dA = A - palette_lab[pi, 1]
                dB = B - palette_lab[pi, 2]
                dist = dL*dL + dA*dA + dB*dB
                if dist < best_dist:
                    best_dist = dist
                    best_idx = pi

            out_keys[y, x] = palette_keys[best_idx]

            # Compute error
            eL = L - palette_lab[best_idx, 0]
            eA = A - palette_lab[best_idx, 1]
            eB = B - palette_lab[best_idx, 2]

            # Floyd–Steinberg error diffusion
            if x + 1 < W:
                img_lab[y, x+1, 0] += eL * (7/16)
                img_lab[y, x+1, 1] += eA * (7/16)
                img_lab[y, x+1, 2] += eB * (7/16)
            if y + 1 < H and x > 0:
                img_lab[y+1, x-1, 0] += eL * (3/16)
                img_lab[y+1, x-1, 1] += eA * (3/16)
                img_lab[y+1, x-1, 2] += eB * (3/16)
            if y + 1 < H:
                img_lab[y+1, x, 0] += eL * (5/16)
                img_lab[y+1, x, 1] += eA * (5/16)
                img_lab[y+1, x, 2] += eB * (5/16)
            if y + 1 < H and x + 1 < W:
                img_lab[y+1, x+1, 0] += eL * (1/16)
                img_lab[y+1, x+1, 1] += eA * (1/16)
                img_lab[y+1, x+1, 2] += eB * (1/16)

    return out_keys

def closest_cc_dither_hq_cpu(image_array):
    """CPU Floyd–Steinberg dithering in Lab space."""
    # Convert palette to LAB
    pal_vals = np.array(list(cc_palette.values()), dtype=np.float32) / 255.0
    pal_lin = srgb_to_linear_np(pal_vals)
    pal_lab = rgb_to_lab_np(pal_lin)
    palette_keys_arr = np.array(list(cc_palette.keys()), dtype=np.int32)

    # Prepare image in LAB
    img = image_array.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    img_lin = srgb_to_linear_np(img)
    img_lab = rgb_to_lab_np(img_lin)

    # Run dithering
    out_keys = _fs_dither_lab(img_lab.copy(), pal_lab, palette_keys_arr)
    return out_keys

# GPU consumer — owns CUDA, batches frames from frame_queue, dithers & writes files
def gpu_consumer(frame_queue: mp.Queue, output_dir: str, target_res, precalc_flag, status_queue: mp.Queue = None):
    # initialize CUDA and move palette/keys to device here (only this process uses CUDA)
    global device, palette, palette_keys
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # palette = cc_palette_colors.to(device)
    # palette_keys = cc_palette_keys.to(device)

    BATCH_MAX = 8   # tune: 4..16 (more reduces IPC but increases latency)
    while True:
        item = frame_queue.get()
        if item is None:
            break
        batch = [item]
        # drain a few without blocking too long
        for _ in range(BATCH_MAX - 1):
            try:
                nxt = frame_queue.get_nowait()
            except Exception:
                break
            if nxt is None:
                # ensure sentinel remains (will break outer loop after batch)
                frame_queue.put(None)
                break
            batch.append(nxt)

        # process batch sequentially in this process (no IPC)
        for idx, frame_rgb in batch:
            try:
                out_path = os.path.join(output_dir, f"frame_{idx:03}.ccframe")
                pil_img = Image.fromarray(frame_rgb)
                convert_image_to_ccframe(img=pil_img, output_path=out_path, target_res=target_res, pre_scaled=precalc_flag)
                if status_queue is not None:
                    status_queue.put(('finished', idx))
            except Exception as e:
                # Log error to status queue so main process can print
                if status_queue is not None:
                    status_queue.put(('error', f"{idx}:{e}"))
                else:
                    print("GPU consumer error:", e)

# listener thread which allows subproccesses to print to the command line
def status_listener(status_queue: mp.Queue, total_frames: int, displayProgress: bool = True):
    last_loaded = -1
    while True:
        msg = status_queue.get()
        if msg is None:
            break
        typ, payload = msg
        if typ == 'finished':
            idx = payload
            if displayProgress:
                print(f"Finished frame {idx + 1} / {total_frames}\033[K", end='\r', flush=True)
        elif typ == 'loaded':
            idx = payload
            # reduce print frequency for loaded messages
            if displayProgress and (idx % 10 == 0):
                print(f"Loading frame into memory {idx + 1} / {total_frames}\033[K", end='\r', flush=True)
        elif typ == 'error':
            print("Worker error:", payload)

# Create a lookup table mapping ComputerCraft color keys to their hex digit (in ASCII)
lookup = np.full(32769, ord("f"), dtype=np.uint8)  
for k, v in hex_map.items(): 
    lookup[k] = ord(v)

# QUERIES
def is_video_file(filepath):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    ext = os.path.splitext(filepath)[1].lower()
    return ext in video_extensions
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
    r_frame_rate = info['streams'][0]['r_frame_rate']  # like '30000/1001'
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
def is_image_file(filepath):
    exclude_formats = ["GIF"]
    try:
        with Image.open(filepath) as img:
            img.verify()
            if img.format.lower() in [fmt.lower() for fmt in exclude_formats]:
                return False
        return True
    except Exception:
        return False
def is_gif(filepath):
    try:
        with Image.open(filepath) as img:
            return img.format == "GIF"
    except Exception:
        return False
def get_allowed_cores(num_cores, skip_core0=True):
    cores = list(range(psutil.cpu_count(logical=True)))
    if skip_core0 and 0 in cores:
        cores.remove(0)
    return cores[:num_cores]
def calculate_resolution(wh,total_res):
    width, height =  wh
    image_aspect = (width * 1.5) / height

    if image_aspect > aspect[0] / aspect[1]:
        calc_width = math.floor(total_res[0])
        calc_height = math.floor(total_res[0] / image_aspect)
    else:
        calc_height = math.floor(total_res[1])
        calc_width = math.floor(total_res[1] * image_aspect)

    calc = (calc_width// 2 * 2, calc_height// 2 * 2)
    print("Calculated resolution to match aspect ratio:",calc)
    return calc

# Image Array to CCFRAME w/o return, is called by mp4 and gif to frames for individual frame handling
def convert_image_to_ccframe(input_path="", img = None, output_path="", target_res=(-1,-1), pre_scaled = False):
    ccname = ""
    if img == None:
        print("Loading image:",input_path)
        img = Image.open(input_path).convert("RGB")
        if output_path == "":
            parts = input_path.split(".")
            parts.pop()
            new_filename = ".".join(parts) + ".bmp"
            ccname = ".".join(parts) + ".ccframe"
            output_path = new_filename
    else:
        ccname = output_path

    if not pre_scaled:
        img = img.resize(target_res, Image.Resampling.NEAREST)


    np_img = np.array(img)

    if dither:
        if HQDither:
            cc_matrix = closest_cc_dither_hq_cpu(np_img)
        else:
            cc_matrix = closest_cc_dither(np_img)
    else:
        cc_matrix = closest_cc(np_img)

    lua_lines = ["return {"]

    for row in cc_matrix:
        bg_line = bytes(lookup[row]).decode("ascii")
        text_line = " " * len(bg_line)
        text_color_line = "0" * len(bg_line)
        lua_lines.append(f'  {{"{text_line}", "{text_color_line}", "{bg_line}"}},')

    lua_lines.append("}")

    with open(ccname, "w") as f:
        f.write("\n".join(lua_lines))

def convert_image_to_ccanim_frame(input_path="", img = None, output_path="", target_res=(-1,-1), pre_scaled = False):
    if not pre_scaled:
        img = img.resize(target_res, Image.Resampling.NEAREST)

    np_img = np.array(img)

    if dither:
        if HQDither:
            cc_matrix = closest_cc_dither_hq_cpu(np_img)
        else:
            cc_matrix = closest_cc_dither(np_img)
    else:
        cc_matrix = closest_cc(np_img)

    lua_frame_lines = []
    for row in cc_matrix:
        bg_line = bytes(lookup[row]).decode("ascii")
        text_line = " " * len(bg_line)
        text_color_line = "0" * len(bg_line)
        lua_frame_lines.append(f'{{"{text_line}", "{text_color_line}", "{bg_line}"}}')

    return "{ " + ", ".join(lua_frame_lines) + " }"

# Video to CCFRAME
def mp4_to_frames(input_path,targetFPS,total_res):
    
    input_path = input_path

    output_dir = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper()
    print("Working Dir",output_dir)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    width,height = get_video_resolution(input_path)

    target_res=calculate_resolution((width,height),total_res)

    current_fps = get_fps(input_path)
    targetFPS = min(current_fps,targetFPS)

    print(f"Resampling {current_fps} to {targetFPS} FPS") 

    if not legacyGPUSupport:
        (
            ffmpeg
            .input(input_path, hwaccel='cuda', hwaccel_device=0)
            .filter('fps', fps=targetFPS, round='up')
            .filter('scale', width=target_res[0], height=target_res[1])
            .output(output_dir + "/TMP.mp4", vcodec='h264_nvenc')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )
    else:
        (
        ffmpeg
            .input(input_path, hwaccel='cuda', hwaccel_device=0)
            .filter('fps', fps=targetFPS, round='up')
            .filter('scale', width=target_res[0], height=target_res[1])
            .output(output_dir + "/TMP.mp4", vcodec='libx264')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )
    
    input_path = output_dir + "/TMP.mp4"


    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    maxCache = queueSize if OverideQueueSize else total_frames
    global frameCalcStart

    

    # Prepare multiprocessing pool (CPU producers)
    allowed_cores = get_allowed_cores(poolSize, skip_core0=True)
    producer_pool = Pool(processes=len(allowed_cores), initializer=init_worker, initargs=(allowed_cores,))

    # frame queue and status queue
    frame_queue = mp.Queue(maxsize=max(64, 8 * len(allowed_cores)))  # larger queue to decouple producers and consumer
    status_queue = mp.Queue()

    # start status listener thread in main process
    listener_t = threading.Thread(target=status_listener, args=(status_queue, total_frames, displayProgress), daemon=True)
    listener_t.start()

    # start gpu consumer process (single process that owns CUDA)
    gpu_proc = mp.Process(target=gpu_consumer, args=(frame_queue, output_dir, target_res, True, status_queue))
    gpu_proc.start()

    # Prepare multiprocessing pool
    allowed_cores = get_allowed_cores(poolSize, skip_core0=True)
    try:
        while True:
            frames_batch = []
            for _ in range(maxCache):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append((frame_number, frame_rgb, output_dir, aspect, not careAboutExtraEdgePixels, total_frames))
                # only print occasional loading messages here to avoid I/O overhead
                print(f"Loading frame into memory {frame_number + 1} / {total_frames}\033[K", end='\r', flush=True)
                frame_number += 1

            if not frames_batch:
                break

            if frameCalcStart == 0:
                            frameCalcStart = time.time()
            # Map to CPU pool to do (minimal) preprocessing — returns (idx, frame_rgb)
            # chunksize reduces scheduling overhead — tune if needed
            
            for result in producer_pool.imap_unordered(itemgetter(0, 1), frames_batch, chunksize=4):
                frame_queue.put(result)

            gc.collect()

    finally:
        # all frames queued; shut down producers and consumers cleanly
        producer_pool.close()
        producer_pool.join()

        # signal consumer to stop
        frame_queue.put(None)
        gpu_proc.join()

        # stop listener thread
        status_queue.put(None)
        listener_t.join()

        cap.release()

        # cleanup
        try:
            os.remove(input_path)
        except OSError:
            pass

    # torch.cuda.empty_cache()

    cap.release()

    try:
        os.remove(input_path)
    except OSError:
        pass

def gpu_consumer_collect(queue, target_res, status_queue, animation_frames):
    while True:
        item = queue.get()
        if item is None:
            break
        idx, frame_rgb = item
        lua_string = convert_image_to_ccanim_frame(img=Image.fromarray(frame_rgb), output_path="", target_res=target_res, pre_scaled=True)
        animation_frames.append(lua_string)
        status_queue.put(('finished', idx+1))

def mp4_to_ccanim(input_path, targetFPS, total_res):
    output_name = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper() + ".ccanim"
    print("Output file:", output_name)

    # Resample and scale via FFmpeg
    width, height = get_video_resolution(input_path)
    target_res = calculate_resolution((width, height), total_res)
    current_fps = get_fps(input_path)
    targetFPS = min(current_fps, targetFPS)

    print(f"Resampling {current_fps} to {targetFPS} FPS") 

    if not legacyGPUSupport:
        (
            ffmpeg
            .input(input_path, hwaccel='cuda', hwaccel_device=0)
            .filter('fps', fps=targetFPS, round='up')
            .filter('scale', width=target_res[0], height=target_res[1])
            .output("TMP.mp4", vcodec='h264_nvenc')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )
    else:
        (
            ffmpeg
            .input(input_path, hwaccel='cuda', hwaccel_device=0)
            .filter('fps', fps=targetFPS, round='up')
            .filter('scale', width=target_res[0], height=target_res[1])
            .output("TMP.mp4", vcodec='libx264')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )

    input_path = "TMP.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    maxCache = queueSize if OverideQueueSize else total_frames
    global frameCalcStart

    # Store final animation frames here
    # animation_frames = []

    # Prepare multiprocessing pool (CPU producers)
    allowed_cores = get_allowed_cores(poolSize, skip_core0=True)
    producer_pool = Pool(processes=len(allowed_cores), initializer=init_worker, initargs=(allowed_cores,))

    # frame queue and status queue
    frame_queue = mp.Queue(maxsize=max(64, 8 * len(allowed_cores)))
    status_queue = mp.Queue()

    # start status listener thread
    listener_t = threading.Thread(target=status_listener, args=(status_queue, total_frames, displayProgress), daemon=True)
    listener_t.start()

    manager = Manager()
    animation_frames = manager.list()

    gpu_proc = mp.Process(target=gpu_consumer_collect, args=(frame_queue, target_res, status_queue, animation_frames))
    gpu_proc.start()

    try:
        while True:
            frames_batch = []
            for _ in range(maxCache):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append((frame_number, frame_rgb))
                print(f"Loading frame into memory {frame_number + 1} / {total_frames}\033[K", end='\r', flush=True)
                frame_number += 1

            if not frames_batch:
                break

            if frameCalcStart == 0:
                frameCalcStart = time.time()

            for result in producer_pool.imap_unordered(itemgetter(0, 1), frames_batch, chunksize=4):
                frame_queue.put(result)

            gc.collect()

    finally:
        producer_pool.close()
        producer_pool.join()
        frame_queue.put(None)
        gpu_proc.join()
        status_queue.put(None)
        listener_t.join()
        cap.release()

        try:
            os.remove(input_path)
        except OSError:
            pass
    
    print("Compressing and Saving. This can take a while!")

    lua_data = "return {\n" + "".join(animation_frames) + "}\n"
    compressed = zlib.compress(lua_data.encode("utf-8"), level=9)
    b64 = base64.b64encode(compressed).decode("ascii")
    wrapped = "\n".join(textwrap.wrap(b64, 200))

    with open(output_name, 'w', encoding='utf-8') as f:
        f.write(f'return "{wrapped}"\n')

    print(f"Animation saved to {output_name}")


# Gif to frames
def gif_to_frames(input_path,total_res):
    output_dir = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper()
    print("Working Dir",output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    with Image.open(input_path) as img:
        frame_count = img.n_frames
        print(f"Total frames: {frame_count}")
        target_res = (0,0)
        for frame_number in range(frame_count):
            if frame_number == 0:
                target_res = calculate_resolution(img.size,total_res)
            img.seek(frame_number)
            frame = img.copy()
            duration = img.info.get("duration",50)

            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            frame_path = os.path.join(output_dir, f"frame_{frame_number:03}_{duration:2}.ccframe")
            convert_image_to_ccframe(img = frame,output_path=frame_path,target_res=target_res)
            print(f"Processing frame {frame_number + 1} / {frame_count}", end='\r', flush=True)

# Main Function
def main(input_path,totalMonitors,aspect,scale,targetFPS):
    total_res = (math.floor((round(21.33266129032258 * (totalMonitors[0]/aspect[0]) - 6.645161290322656)*aspect[0])/(scale*2)), math.floor((round(14.222324046920821 * (totalMonitors[1]/aspect[1]) - 4.449596774193615)*aspect[1])/(scale*2)))
    print("In game resolution to match:", total_res)

    if is_image_file(input_path):
        with Image.open(input_path) as img:
            convert_image_to_ccframe(input_path=input_path,target_res=calculate_resolution(img.size,total_res))
    elif is_gif(input_path):
        gif_to_frames(input_path,total_res)
    elif is_video_file(input_path):
        mp4_to_ccanim(input_path,targetFPS,total_res)
    else:
        print("Unsupported file type.")
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    print(f"Time for frames: {end - frameCalcStart:.2f} seconds")

    output_dir = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper()
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Ensure file still exists
                total_size += os.path.getsize(fp)
    print(f"Final size: {total_size/ (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    with cProfile.Profile() as pr:
        if len(sys.argv) < 7:
            print("Usage: python convert.py <media_file (Any* Image / GIF / mp4, avi, mov, mkv)> <Block Width> <Block Height> <Cell # X> <Cell # Y> <Text Scale> <FPS <= 20 (Optional)>")
            sys.exit(1)

        input_file = sys.argv[1]
        totalMonitors = (int(sys.argv[2]),int(sys.argv[3]))
        aspect = (int(sys.argv[4]),int(sys.argv[5]))
        scale = float(sys.argv[6])

        if len(sys.argv) > 7:
            if float(sys.argv[7]) > 20:
                print("\033[91mREQUESTED FPS TOO HIGH, SETTING TO 20\033[0m")
            targetFPS = min(float(sys.argv[7]),20)
        else:
            targetFPS = 20

        if not os.path.exists(input_file):
            print(f"Error: file not found: {input_file}")
            sys.exit(1)
        
        if profile:
            pr = cProfile.Profile()
            pr.enable()

            cProfile.run('main(input_file,totalMonitors,aspect,scale,targetFPS)', 'profile_data.prof')

            pr.disable()
            pr.dump_stats('profile_data.prof')
        else:
            main(input_file,totalMonitors,aspect,scale,targetFPS)