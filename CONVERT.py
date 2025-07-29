from PIL import Image
import sys
import os
import math
import numpy as np
import torch
import time
import cv2
import ffmpeg
import json
import subprocess

aspect = (3,3)
targetFPS = 20
scale = 0.5
total_res = (math.floor(188.5 / scale) * aspect[0], math.floor(126 / scale) * aspect[1])

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

times = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("\033[91mWARNING: Python Version Mismatch or Not Installed Properly. Use python 3.11!\033[0m")
cc_palette_keys = torch.tensor(list(cc_palette.keys()))
cc_palette_colors = torch.tensor(list(cc_palette.values()), dtype=torch.float32) 


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

def get_closest_cc_matrix_torch(image_array):

    img_tensor = torch.from_numpy(image_array).float().to(device).half()

    H, W, _ = img_tensor.shape

    pixels = img_tensor.view(-1, 3)  

    palette = cc_palette_colors.to(device).half()

    diffs = pixels.unsqueeze(1) - palette.unsqueeze(0)  

    dists = torch.cdist(pixels, palette)
    nearest_idxs = torch.argmin(dists, dim=1)

    keys = cc_palette_keys.to(nearest_idxs.device)[nearest_idxs].to("cpu")

    return keys.view(H, W).numpy()

def closest_cc_color(rgb):
    r, g, b = rgb
    closest = min(cc_palette.items(),
                  key=lambda kv: (kv[1][0]-r)**2 + (kv[1][1]-g)**2 + (kv[1][2]-b)**2)
    return closest[0]

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

def is_video_file(filepath):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    ext = os.path.splitext(filepath)[1].lower()
    return ext in video_extensions

def gif_to_frames(input_path):
    output_dir = "TMP"
    os.makedirs(output_dir, exist_ok=True)

    with Image.open(input_path) as img:
        frame_count = img.n_frames
        print(f"Total frames: {frame_count}")

        for frame_number in range(frame_count):
            img.seek(frame_number)
            frame = img.copy()
            duration = img.info.get("duration",50)

            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            frame_path = os.path.join(output_dir, f"frame_{frame_number:03}_{duration:2}.ccframe")
            convert_image_to_bmp(img = frame,output_path=frame_path)
            print(f"Processing frame {frame_number + 1} / {frame_count}", end='\r', flush=True)

def convert_image_to_bmp(input_path="", img = None, output_path=""):
    ccname = ""
    if img == None:
        print(input_path)
        img = Image.open(input_path).convert("RGB")
        if output_path == "":
            parts = input_path.split(".")
            parts.pop()
            new_filename = ".".join(parts) + ".bmp"
            ccname = ".".join(parts) + ".ccframe"
            output_path = new_filename
    else:
        ccname = output_path
    
    width, height = img.size
    calc_width, calc_height = 0,0


    target_aspect = aspect[0] / aspect[1]
    image_aspect = (width * 1.5) / height

    if image_aspect > target_aspect:
        calc_width = total_res[0]
        calc_height = total_res[0] / image_aspect
    else:
        calc_height = total_res[1]
        calc_width = total_res[1] * image_aspect
    
    calc = (math.floor(calc_width),math.floor(calc_height))
    img = img.resize(calc, Image.Resampling.NEAREST)


    np_img = np.array(img)
    cc_matrix = get_closest_cc_matrix_torch(np_img)

    lua_lines = ["return {"]

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

    for row in cc_matrix:
        bg_line = "".join(hex_map.get(px, "f") for px in row)
        text_line = " " * len(bg_line)
        text_color_line = "0" * len(bg_line)
        lua_lines.append(f'  {{"{text_line}", "{text_color_line}", "{bg_line}"}},')

    lua_lines.append("}")

    with open(ccname, "w") as f:
        f.write("\n".join(lua_lines))


def mp4_to_frames(input_path):
    output_dir = "TMP"
    os.makedirs(output_dir, exist_ok=True)

    current_fps = get_fps(input_path)
    print(f"Input FPS: {current_fps}")
    if current_fps > targetFPS:
        print(f"Resampling to a lower FPS")
        (
            ffmpeg
            .input(input_path)
            .filter('fps', fps=targetFPS, round='up')
            .output(output_dir + "\\TMP.mp4")
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )
        input_path = output_dir + "\\TMP.mp4"

    cap = cv2.VideoCapture(input_path)
    frame_number = 0
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame is in BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        frame_path = os.path.join(output_dir, f"frame_{frame_number:03}.ccframe")
        convert_image_to_bmp(img=pil_img, output_path=frame_path)
        print(f"Processing frame {frame_number + 1} / {total_frames}", end='\r', flush=True)

        frame_number += 1

    cap.release()
    if current_fps > 20:
        if os.path.exists(input_path):
            os.remove(input_path)

def main(input_path):
    start = time.time()
    if is_image_file(input_path):
        convert_image_to_bmp(input_path)
    elif is_gif(input_path):
        gif_to_frames(input_path)
    elif is_video_file(input_path):
        mp4_to_frames(input_path)
    else:
        print("Unsupported file type.")
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <image_file> [output.bmp]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else ""

    if not os.path.exists(input_file):
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    main(input_file)
