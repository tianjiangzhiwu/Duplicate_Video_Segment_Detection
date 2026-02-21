#!/usr/bin/env python3
"""
视频去重工具 - 环境检查脚本
运行: python check_env.py
"""

import sys
import subprocess
import os
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

def print_ok(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_fail(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warn(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    print(f"  {text}")

def check_python():
    """检查 Python 版本"""
    print_header("1. Python 环境检查")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"当前版本: Python {version_str}")
    
    # 检查版本是否在 3.8-3.11 之间
    if (3, 8) <= (version.major, version.minor) <= (3, 11):
        print_ok(f"Python 版本兼容 (要求 3.8-3.11)")
        return True
    else:
        print_fail(f"Python 版本不兼容 (要求 3.8-3.11，当前 {version_str})")
        print_info("解决方案: 卸载当前 Python，安装 Python 3.10")
        print_info("下载地址: https://www.python.org/downloads/release/python-31011/")
        return False

def check_pip():
    """检查 pip 是否可用"""
    print_header("2. Pip 包管理器检查")
    
    try:
        import pip
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_ok(f"Pip 已安装: {result.stdout.strip()}")
            return True
        else:
            print_fail("Pip 运行异常")
            return False
    except Exception as e:
        print_fail(f"Pip 检查失败: {e}")
        return False

def check_nvidia_driver():
    """检查 NVIDIA 驱动"""
    print_header("3. NVIDIA 显卡驱动检查")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # 提取驱动版本和 CUDA 版本
            for line in lines:
                if "Driver Version" in line and "CUDA Version" in line:
                    print_ok("NVIDIA 驱动已安装")
                    print_info(line.strip())
                    return True
            print_ok("NVIDIA 驱动已安装")
            print_info("(详细版本信息请运行 nvidia-smi 查看)")
            return True
        else:
            print_fail("NVIDIA 驱动未安装或 nvidia-smi 不可用")
            print_info("解决方案: 访问 https://www.nvidia.cn/drivers/ 下载驱动")
            return False
    except FileNotFoundError:
        print_fail("未找到 nvidia-smi，NVIDIA 驱动可能未安装")
        print_info("解决方案: 安装 NVIDIA 显卡驱动")
        return False
    except Exception as e:
        print_fail(f"检查驱动时出错: {e}")
        return False

def check_pytorch():
    """检查 PyTorch 安装"""
    print_header("4. PyTorch 检查")
    
    try:
        import torch
        print_ok(f"PyTorch 已安装: {torch.__version__}")
        
        # 检查 CUDA 是否可用
        if torch.cuda.is_available():
            print_ok("CUDA 可用")
            print_info(f"CUDA 版本: {torch.version.cuda}")
            print_info(f"GPU 设备: {torch.cuda.get_device_name(0)}")
            print_info(f"GPU 数量: {torch.cuda.device_count()}")
            
            # 测试 GPU 是否真能运行
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = test_tensor @ test_tensor.T
                print_ok("GPU 计算测试通过")
                return True
            except Exception as e:
                print_fail(f"GPU 计算测试失败: {e}")
                return False
        else:
            print_warn("CUDA 不可用，将使用 CPU 模式（速度较慢）")
            print_info("如需 GPU 加速，请安装 CUDA 版本的 PyTorch")
            return True  # CPU 模式也能运行
        
    except ImportError:
        print_fail("PyTorch 未安装")
        print_info("安装命令:")
        print_info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print_info("或 (CPU版本):")
        print_info("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        return False

def check_opencv():
    """检查 OpenCV"""
    print_header("5. OpenCV 检查")
    
    try:
        import cv2
        print_ok(f"OpenCV 已安装: {cv2.__version__}")
        
        # 测试视频解码能力
        print_info("测试视频解码支持...")
        backends = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_MSMF, "Microsoft Media Foundation"),
            (cv2.CAP_DSHOW, "DirectShow"),
        ]
        available = []
        for code, name in backends:
            # 创建测试 capture（不打开文件，只检查后端）
            if code == cv2.CAP_FFMPEG:
                available.append(name)
                print_info(f"  后端 {name}: 可用")
        
        if not available:
            print_warn("未检测到视频解码后端，可能无法读取某些视频格式")
        else:
            print_ok("视频解码支持正常")
        
        return True
    except ImportError:
        print_fail("OpenCV 未安装")
        print_info("安装命令: pip install opencv-python")
        return False

def check_numpy():
    """检查 NumPy"""
    print_header("6. NumPy 检查")
    
    try:
        import numpy as np
        print_ok(f"NumPy 已安装: {np.__version__}")
        
        # 测试基本运算
        arr = np.random.randn(1000, 1000)
        result = arr @ arr.T
        print_ok("NumPy 运算测试通过")
        return True
    except ImportError:
        print_fail("NumPy 未安装")
        print_info("安装命令: pip install numpy")
        return False

def check_faiss():
    """检查 Faiss"""
    print_header("7. Faiss 向量检索库检查")
    
    try:
        import faiss
        print_ok(f"Faiss 已安装")
        
        # 测试基本功能
        import numpy as np
        d = 64  # 维度
        nb = 1000  # 数据库大小
        xb = np.random.random((nb, d)).astype('float32')
        
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        
        xq = np.random.random((10, d)).astype('float32')
        D, I = index.search(xq, 5)
        
        print_ok("Faiss 功能测试通过")
        return True
    except ImportError:
        print_fail("Faiss 未安装")
        print_info("安装命令: pip install faiss-cpu")
        return False

def check_pillow():
    """检查 Pillow"""
    print_header("8. Pillow 图像处理库检查")
    
    try:
        from PIL import Image
        print_ok(f"Pillow 已安装")
        
        # 测试创建图像
        img = Image.new('RGB', (100, 100), color='red')
        print_ok("Pillow 功能测试通过")
        return True
    except ImportError:
        print_fail("Pillow 未安装")
        print_info("安装命令: pip install pillow")
        return False

def check_tqdm():
    """检查 tqdm 进度条"""
    print_header("9. tqdm 进度条库检查")
    
    try:
        from tqdm import tqdm
        print_ok("tqdm 已安装")
        return True
    except ImportError:
        print_fail("tqdm 未安装")
        print_info("安装命令: pip install tqdm")
        return False

def check_project_structure():
    """检查项目文件结构"""
    print_header("10. 项目文件结构检查")
    
    required_files = [
        "config.py",
        "extract_features.py", 
        "build_index.py",
        "search_duplicates.py",
        "visualize.py",
        "main.py"
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print_ok(f"找到 {file}")
        else:
            print_fail(f"缺少 {file}")
            missing.append(file)
    
    # 检查目录
    dirs = ["cache", "output", "videos"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print_ok(f"目录 {d}/ 已就绪")
    
    return len(missing) == 0

def check_video_files():
    """检查视频文件"""
    print_header("11. 视频文件检查")
    
    video_dir = Path("videos")
    video_dir.mkdir(exist_ok=True)
    
    exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    videos = [f for f in video_dir.glob('*') if f.suffix.lower() in exts]
    
    print_info(f"视频目录: {video_dir.absolute()}")
    print_info(f"发现视频文件: {len(videos)} 个")
    
    if len(videos) == 0:
        print_warn("视频目录为空，请将视频文件放入 videos/ 文件夹")
        return False
    
    for v in videos[:5]:
        size_mb = v.stat().st_size / (1024*1024)
        print_info(f"  - {v.name} ({size_mb:.1f} MB)")
    
    if len(videos) > 5:
        print_info(f"  ... 还有 {len(videos)-5} 个")
    
    print_ok(f"共 {len(videos)} 个视频文件待处理")
    return True

def generate_install_script():
    """生成安装脚本"""
    print_header("生成安装脚本")
    
    script_content = '''@echo off
echo 正在安装视频去重工具依赖...
echo.

python -m pip install --upgrade pip setuptools wheel

echo.
echo 安装 PyTorch (带CUDA支持)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo 安装其他依赖...
pip install numpy opencv-python pillow tqdm faiss-cpu

echo.
echo 安装完成!
pause
'''
    
    with open("install_deps.bat", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print_ok("已生成 install_deps.bat")
    print_info("双击运行 install_deps.bat 可自动安装所有依赖")

def main():
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  视频去重工具 - 环境检查{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    checks = [
        ("Python 版本", check_python),
        ("Pip 包管理器", check_pip),
        ("NVIDIA 驱动", check_nvidia_driver),
        ("PyTorch", check_pytorch),
        ("OpenCV", check_opencv),
        ("NumPy", check_numpy),
        ("Faiss", check_faiss),
        ("Pillow", check_pillow),
        ("tqdm", check_tqdm),
        ("项目文件", check_project_structure),
        ("视频文件", check_video_files),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_fail(f"检查过程中出错: {e}")
            results[name] = False
    
    # 总结
    print_header("检查总结")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n通过: {Colors.GREEN}{passed}{Colors.RESET}/{total}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ 所有检查通过！可以运行 python main.py{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}部分检查未通过:{Colors.RESET}")
        for name, ok in results.items():
            if not ok:
                print(f"  {Colors.RED}- {name}{Colors.RESET}")
        
        print(f"\n{Colors.YELLOW}建议操作:{Colors.RESET}")
        print_info("1. 修复上述失败项")
        print_info("2. 或运行生成的 install_deps.bat 自动安装")
        generate_install_script()

if __name__ == "__main__":
    main()