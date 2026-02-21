from pathlib import Path
import torch

# ==================== 手动配置区域 ====================
# 修改这里：填入你的视频文件夹路径（支持中文、空格）
# 示例：
#   Windows: VIDEO_DIR = Path(r"D:\我的视频\素材")
#   Mac/Linux: VIDEO_DIR = Path("/home/user/videos")

VIDEO_DIR = Path(r"D:\林一")  # <-- 修改这一行

# ===================================================

# 其他配置（一般不需要改）
CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")

# 处理参数
SAMPLE_INTERVAL = 1.0          # 每秒采样1帧（越小越精确，越慢）
MIN_DUPLICATE_DURATION = 5.0  # 最小重复片段长度（秒）
FRAME_RESIZE = (224, 224)      # 帧缩放尺寸

# 设备自动检测
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.88    # 相似度阈值（0.85-0.95）
TOP_K = 50                     # 检索最近邻数量

# 创建目录
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 启动信息
print(f"=" * 60)
print(f"视频去重工具配置")
print(f"=" * 60)
print(f"视频目录: {VIDEO_DIR.absolute()}")
print(f"缓存目录: {CACHE_DIR.absolute()}")
print(f"输出目录: {OUTPUT_DIR.absolute()}")
print(f"使用设备: {DEVICE}")
print(f"采样间隔: {SAMPLE_INTERVAL}秒")
print(f"最小重复时长: {MIN_DUPLICATE_DURATION}秒")
print(f"=" * 60)

# 验证视频目录
if not VIDEO_DIR.exists():
    print(f"\n⚠️ 警告: 视频目录不存在: {VIDEO_DIR}")
    print("请修改 config.py 中的 VIDEO_DIR 路径")
    print(f"或创建目录: mkdir \"{VIDEO_DIR}\"")
elif not any(VIDEO_DIR.iterdir()):
    print(f"\n⚠️ 警告: 视频目录为空: {VIDEO_DIR}")
    print("请将视频文件放入该目录")