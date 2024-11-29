from moviepy.editor import VideoFileClip, concatenate_videoclips
os.chdir('/content/drive/MyDrive/Colab Notebooks')

"""#Make frames from video"""

def concatenate_videos(video_files, output_file):
    clips = [VideoFileClip(video) for video in video_files]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264")
directory = 'indonesian'
video_files = [os.path.join(directory, video) for video in os.listdir(directory)]
output_file = os.path.join(directory, 'indonesians.mp4')
concatenate_videos(video_files, output_file)

def save_frame_range(video_path, start_frame, stop_frame, step_frame,
                     dir_path, basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    frames = []

    # First, extract all the frames you want to save
    for n in range(start_frame, stop_frame, step_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    random.shuffle(frames)  # Shuffle the frames

    # Save the shuffled frames
    for idx, frame in enumerate(frames):
        cv2.imwrite('{}_{}.{}'.format(base_path, str(idx).zfill(digit), ext), frame)

save_frame_range('movie/indian.mp4',
                 0, 7000,70,
                 'result', '')

"""#Remove bcgr"""

import io
input_dir = 'result'
output_dir = 'removed'
# 入力ディレクトリ内の画像ファイルのリストを取得
paths = os.listdir(input_dir)
# random.shuffle(paths)
os.makedirs(output_dir, exist_ok=True)
for idx, path in enumerate(paths):
    input_path = os.path.join(input_dir, path)

    # 新しいファイル名を連番にする
    # new_filename = f"image_{idx:05}.png"
    output_path = os.path.join(output_dir, path)

    # 画像の読み込み
    input_image = Image.open(input_path)

    # 背景除去
    output_image = remove(input_image)

    # 背景除去した画像の保存
    output_image.save(output_path, 'png')

"""#Trimming image"""

input_dir = 'removed'
output_dir = 'trimmed'

# 出力ディレクトリの作成
os.makedirs(output_dir, exist_ok=True)

# 入力ディレクトリ内の画像ファイルのリストを取得
paths = os.listdir(input_dir)
# random.shuffle(paths)
# トリミング領域の指定 (left, top, right, bottom)
crop_rectangle = (0, 700, 2160, 3140)

# 各画像に対して処理
for idx, path in enumerate(paths):
    input_path = os.path.join(input_dir, path)
    new_filename = f"{idx:05}.png"
    output_path = os.path.join(output_dir, new_filename)
    # 画像の読み込み
    input_image = Image.open(input_path)

    # 画像のトリミング
    cropped_image = input_image.crop(crop_rectangle)

    # RGBAモードへの変換（必要に応じて）
    if cropped_image.mode != 'RGBA':
        cropped_image = cropped_image.convert('RGBA')

    # # 背景除去
    # output_image = remove(cropped_image)

    # 背景除去した画像の保存
    cropped_image.save(output_path, 'PNG')

"rename"
# image = sorted(glob(FromImgName))
#imgフォルダ内の画像名をまとめて取得
files = os.listdir(ImgName)
out = '/content/drive/MyDrive/Colab Notebooks/te/0/undistorted/images'
print(files)
# for idx,file in zip(range(len(files)),files):
#     img_path = os.path.join(ImgName, file)
#     print(idx)
#     img = Image.open(img_path)
#     new_filename = f"{idx:05}.png"
#     output_path = os.path.join(out, new_filename)
#     # RGBAモードの画像をRGBモードに変換
#     # if img.mode == 'RGBA':
#     #     img = img.convert('RGBA')

#     # img_resize = img.resize((400, 400))
#     img.save(output_path)

print(len(files))

"resize"

"""#Resize"""

ImgName = 'NeuRBF1/data/nerf_synthetic/indians/train/'

# image = sorted(glob(FromImgName))
#imgフォルダ内の画像名をまとめて取得
files = os.listdir(ImgName)
out = 'NeuRBF1/data/nerf_synthetic/indians/train/'

for idx,file in enumerate(files):
    img_path = os.path.join(ImgName, file)

    img = Image.open(img_path)
    # new_filename = f"{idx+85:05}.png"
    output_path = os.path.join(out, file)
    # RGBAモードの画像をRGBモードに変換
    # if img.mode == 'RGBA':
    #     img = img.convert('RGBA')

    img_resize = img.resize((800, 800))
    img_resize.save(output_path)

"""#File transport

"""

img_file_png = sorted(glob("/content/drive/MyDrive/Colab Notebooks/te/0/undistorted/images/*"))
for filename in img_file_png:
    shutil.move(filename, 'NeuRBF1/data/nerf_synthetic/indians/test/')

for i in range(94):
    print(out['frames'][i]['file_path'])