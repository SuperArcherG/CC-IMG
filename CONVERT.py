from PIL import Image
from multiprocessing import Pool, cpu_count
import sys, os, math, torch, time, cv2, ffmpeg, json, subprocess, numpy as np, re, shutil, gc


import cProfile

start = time.time()
frameCalcStart = 0

# # # PERFORMANCE SETTINGS
poolSize = 16
OverideQueueSize = False
queueSize = 128
careAboutExtraEdgePixels = False
profile = False
displayProgress = True

# total_res = (math.floor(188.5 / scale) * aspect[0], math.floor(126 / scale) * aspect[1])

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("\033[91mWARNING: Python Version Mismatch or Not Installed Properly. Use python 3.11!\033[0m")

cc_palette_keys = torch.tensor(list(cc_palette.keys()))
cc_palette_colors = torch.tensor(list(cc_palette.values()), dtype=torch.float32) / 255.0

# cc_palette_colors = cc_palette_colors.to(torch.float32)

## TOP CONTENDER float8_e5m2

palette = cc_palette_colors.to(device)
palette_keys = cc_palette_keys.to(device)

@torch.no_grad()
def closest_cc(image_array):

    img_tensor = torch.from_numpy(image_array).to(device).div(255)

    H, W, _ = img_tensor.shape # Height, Width, Discard Color

    pixels = img_tensor.view(-1, 3) # Flattens the image tensor from (H, W, 3) to (H*W, 3), making a 2D tensor where each row is one pixel’s RGB.

    dists = torch.cdist(pixels, palette)

    nearest_idxs = torch.argmin(dists, dim=1)

    keys = palette_keys[nearest_idxs].to("cpu")
    return keys.view(H, W).numpy()

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



def gif_to_frames(input_path,total_res):
    output_dir = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper()
    print("Working Dir",output_dir)
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
            convert_image_to_ccframe(img = frame,output_path=frame_path,total_res=total_res)
            print(f"Processing frame {frame_number + 1} / {frame_count}", end='\r', flush=True)

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

# Makes a lookup table for every possible color

lookup = np.full(32769, ord("f"), dtype=np.uint8)  
for k, v in hex_map.items(): 
    lookup[k] = ord(v)

def process_frame(args):
        idx, frame_data, total_res, output_dir, aspect, precalc = args
        pil_img = Image.fromarray(frame_data)
        convert_image_to_ccframe(
            img=pil_img,
            output_path=os.path.join(output_dir, f"frame_{idx:03}.ccframe"),
            total_res=total_res,
            aspect=aspect,
            precalc=precalc
        )
        return idx

def convert_image_to_ccframe(input_path="", img = None, output_path="", total_res=(-1,-1), aspect=(-1,-1), precalc = False):
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

    if not precalc:
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
def mp4_to_frames(input_path,targetFPS,total_res):
    output_dir = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper()
    print("Working Dir",output_dir)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    width,height = get_video_resolution(input_path)
    calc_width, calc_height = 0,0
    target_aspect = aspect[0] / aspect[1]
    image_aspect = (width * 1.5) / height

    if image_aspect > target_aspect:
        calc_width = total_res[0]
        calc_height = total_res[0] / image_aspect
    else:
        calc_height = total_res[1]
        calc_width = total_res[1] * image_aspect
    
    calc = (math.floor(calc_width), math.floor(calc_height))
    print("Calculated resolution to match aspect ratio:",calc)
    calc = (calc[0]// 2 * 2, calc[1]// 2 * 2)
    print("Rescaling to:",calc)

    current_fps = get_fps(input_path)
    if current_fps > targetFPS:
        print(f"Resampling {current_fps} to {targetFPS} FPS")  
        (
            ffmpeg
            .input(input_path, hwaccel='cuda', hwaccel_device=0)
            .filter('fps', fps=targetFPS, round='up')
            .filter('scale', width=calc[0], height=calc[1])
            .output(output_dir + "\\TMP.mp4", vcodec='h264_nvenc')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )
        input_path = output_dir + "\\TMP.mp4"
    else:
        print(f"Video at {targetFPS} FPS already")    

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    maxCache = queueSize if OverideQueueSize else total_frames
    global frameCalcStart

    frameCalcStart = time.time()

    # Prepare multiprocessing pool
    with Pool(processes=min(poolSize,cpu_count())) as pool:
        while True:
            frames_batch = []
            for _ in range(maxCache):  # Batch size
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append((frame_number, frame_rgb, total_res, output_dir, aspect, not careAboutExtraEdgePixels))
                if (displayProgress):
                    print(f"Loading frame into memory {frame_number + 1} / {total_frames}", end='\r', flush=True)
                pass
                frame_number += 1

            if not frames_batch:
                break
            
            frameCalcStart = time.time()
            for finishedFrame in list(pool.imap_unordered(process_frame, frames_batch)):
                if (displayProgress):
                    print(f"Finished frame {finishedFrame + 1} / {total_frames}", end='\r', flush=True)

            torch.cuda.empty_cache()
            gc.collect()

    cap.release()

    # try:
    #     os.remove(input_path)
    # except OSError:
    #     pass


def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Ensure file still exists
                total_size += os.path.getsize(fp)
    return total_size
def main(input_path,totalMonitors,aspect,scale,targetFPS):
    total_res = (math.floor((round(21.33266129032258 * (totalMonitors[0]/aspect[0]) - 6.645161290322656)*aspect[0])/(scale*2)), math.floor((round(14.222324046920821 * (totalMonitors[1]/aspect[1]) - 4.449596774193615)*aspect[1])/(scale*2)))
    print("In game resolution to match:", total_res)

    if is_image_file(input_path):
        convert_image_to_ccframe(input_path,total_res)
    elif is_gif(input_path):
        gif_to_frames(input_path,total_res)
    elif is_video_file(input_path):
        mp4_to_frames(input_path,targetFPS,total_res)
    else:
        print("Unsupported file type.")
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    print(f"Time for frames: {end - frameCalcStart:.2f} seconds")

    output_dir = re.sub(r'[^a-zA-Z0-9]', '', os.path.basename(input_path).split('.')[0]).upper()
    print(f"Final size: {get_folder_size(output_dir)/ (1024 * 1024):.2f} MB")
if __name__ == "__main__":
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