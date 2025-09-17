import os
import subprocess
import whisper
import torch
import warnings
from tkinter import Tk, filedialog

# Ẩn cảnh báo FP16 trên CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Mở hộp thoại chọn thư mục
root = Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Chọn thư mục chứa video")

if not folder_path:
    print("❌ Bạn chưa chọn thư mục nào.")
    exit()

# Liệt kê các video trong thư mục
video_extensions = (".mp4", ".mkv", ".avi", ".mov")
video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]

if not video_files:
    print("❌ Không tìm thấy video trong thư mục.")
    exit()

print(f"📂 Thư mục: {folder_path}")
print(f"🎞️ Tìm thấy {len(video_files)} video:")

for idx, v in enumerate(video_files, 1):
    print(f"   {idx}. {v}")

# Check GPU/CPU
if torch.cuda.is_available():
    device = "cuda"
    fp16 = True
    print("\n⚡ Phát hiện GPU CUDA → sử dụng FP16 cho tốc độ nhanh.")
elif torch.backends.mps.is_available():
    device = "mps"
    fp16 = False
    print("\n🍎 Phát hiện Apple Silicon → dùng MPS (FP32).")
else:
    device = "cpu"
    fp16 = False
    print("\n🐢 Không có GPU → chạy CPU (FP32).")

# Load model Whisper
model = whisper.load_model("medium", device=device)

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Trích xuất audio từ video bằng ffmpeg"""
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", audio_path
    ]
    subprocess.run(command, check=True)

def generate_subtitles(video_path, output_srt):
    """Sinh phụ đề từ video"""
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    result = model.transcribe(audio_path, fp16=fp16)
    os.remove(audio_path)

    # Ghi file SRT
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()

            def format_time(t):
                h, m = divmod(t, 3600)
                m, s = divmod(m, 60)
                ms = (s - int(s)) * 1000
                return f"{int(h):02}:{int(m):02}:{int(s):02},{int(ms):03}"

            f.write(f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n")

# Xử lý từng video
for idx, video in enumerate(video_files, 1):
    video_path = os.path.join(folder_path, video)
    output_srt = os.path.splitext(video_path)[0] + ".srt"

    print(f"\n🔄 [{idx}/{len(video_files)}] Đang xử lý: {video} ...")
    try:
        generate_subtitles(video_path, output_srt)
        print(f"✅ Hoàn thành: {output_srt}")
    except Exception as e:
        print(f"❌ Lỗi với {video}: {e}")

print("\n🎉 Xong toàn bộ!")
