#!/usr/bin/env python3
"""
视频重复片段检测工具 - 一键运行
使用方法: python main.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import VIDEO_DIR, CACHE_DIR, OUTPUT_DIR, DEVICE
from extract_features import process_all_videos
from build_index import build_index
from search_duplicates import find_duplicates
from visualize import generate
from generate_deletion_guide import generate_deletion_guide
import json
import os


def setup_config_wizard():
    """配置向导：帮助用户设置视频目录"""
    config_file = Path("config.json")
    
    print("=" * 60)
    print("视频目录配置")
    print("=" * 60)
    print(f"当前视频目录: {VIDEO_DIR.absolute()}")
    print()
    
    # 检查当前目录是否有视频
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    current_videos = [f for f in VIDEO_DIR.glob('*') if f.suffix.lower() in video_exts]
    
    if current_videos:
        print(f"✓ 该目录已有 {len(current_videos)} 个视频文件")
        response = input("是否使用此目录? [Y/n/换路径]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            return True
        if response == '换路径':
            pass  # 继续下面输入新路径
        else:
            print("退出程序")
            sys.exit(0)
    
    # 输入新路径
    print()
    new_path = input("请输入视频目录路径 (支持相对路径如 ./my_videos，或绝对路径如 D:\\Videos): ").strip()
    
    if not new_path:
        print("未输入路径，退出")
        sys.exit(0)
    
    # 解析路径
    new_path = Path(new_path).expanduser().resolve()
    
    # 创建目录（如果不存在）
    if not new_path.exists():
        create = input(f"目录不存在: {new_path}\n是否创建? [Y/n]: ").strip().lower()
        if create in ('', 'y', 'yes'):
            new_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ 已创建目录: {new_path}")
        else:
            print("退出程序")
            sys.exit(0)
    
    # 保存到配置文件
    config_data = {"video_dir": str(new_path)}
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 配置已保存到: {config_file.absolute()}")
    print(f"✓ 视频目录设置为: {new_path}")
    print()
    print("请重新运行程序以加载新配置")
    sys.exit(0)


def check_environment():
    """检查环境和视频文件"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # 检查 PyTorch
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查视频
    print()
    exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    videos = [f for f in VIDEO_DIR.glob('*') if f.suffix.lower() in exts]
    
    print(f"视频目录: {VIDEO_DIR.absolute()}")
    print(f"视频文件: {len(videos)} 个")
    
    if len(videos) == 0:
        print()
        print("⚠ 没有找到视频文件")
        return False
    
    for v in videos[:3]:
        size_mb = v.stat().st_size / (1024*1024)
        print(f"  - {v.name} ({size_mb:.1f} MB)")
    if len(videos) > 3:
        print(f"  ... 还有 {len(videos)-3} 个")
    
    return True


def main():
    # 配置向导
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        setup_config_wizard()
        return
    
    # 检查是否有视频
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    has_videos = any(f.suffix.lower() in video_exts for f in VIDEO_DIR.glob('*'))
    
    if not has_videos:
        print("未找到视频文件，进入配置向导...")
        print("(提示: 以后可以用 python main.py --config 手动配置)")
        setup_config_wizard()
        return
    
    # 正常流程
    if not check_environment():
        return
    
    input("\n按回车开始处理...")
    t0 = time.time()
    
    print("\n[1/4] 提取特征...")
    feats, meta = process_all_videos(VIDEO_DIR)
    if feats is None:
        print("特征提取失败")
        return
    
    print("\n[2/4] 构建索引...")
    build_index(feats)
    
    print("\n[3/4] 检索重复...")
    dups = find_duplicates()
    with open(OUTPUT_DIR / "duplicates.json", 'w', encoding='utf-8') as f:
        json.dump(dups, f, ensure_ascii=False, indent=2)
    
    print("\n[4/4] 生成报告...")
    report = generate()

    print("\n[5/5] 生成删除指南...")
    deletion_guide = generate_deletion_guide()

    print(f"\n{'='*60}")
    print(f"完成! 耗时: {time.time()-t0:.1f}秒")
    print(f"重复片段: {len(dups)} 个")
    print(f"报告: {report}")
    if deletion_guide:
        print(f"删除指南已生成")
    print(f"{'='*60}")

    # 尝试打开报告
    try:
        import webbrowser
        webbrowser.open(f"file:///{report.absolute()}")
    except:
        pass


if __name__ == "__main__":
    main()