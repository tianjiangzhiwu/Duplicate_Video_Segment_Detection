import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from torchvision import models, transforms
from PIL import Image
import hashlib
import os

from config import *


class FeatureExtractor:
    def __init__(self):
        self.device = torch.device(DEVICE)
        print(f"使用设备: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def _load_model(self):
        print("加载 ResNet50 模型...")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval().to(self.device)
        return model
    
    @torch.no_grad()
    def extract(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        feature = self.model(tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(feature)
        return (feature / norm).astype('float32') if norm > 0 else feature.astype('float32')


def get_video_hash(filepath):
    stat = os.stat(filepath)
    return hashlib.md5(f"{filepath}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()


def crop_black_bars(frame, threshold=10):
    if frame is None or frame.size == 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return frame
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    margin = 2
    return frame[max(0, ymin-margin):min(frame.shape[0], ymax+margin),
                 max(0, xmin-margin):min(frame.shape[1], xmax+margin)]


def extract_video_features(video_path, extractor):
    print(f"\n处理: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  [错误] 无法打开")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"  时长: {duration:.1f}s, 帧数: {total_frames}")
    
    features, timestamps = [], []
    sample_n = max(1, int(fps * SAMPLE_INTERVAL))
    
    pbar = tqdm(total=total_frames, desc="  提取", unit="帧", ncols=70)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_n == 0:
            frame = crop_black_bars(frame)
            frame = cv2.resize(frame, FRAME_RESIZE)
            features.append(extractor.extract(frame))
            timestamps.append(frame_idx / fps)
        
        frame_idx += 1
        pbar.update(1)
        
        if frame_idx % 1000 == 0 and DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    cap.release()
    pbar.close()
    
    return {
        'features': np.array(features),
        'timestamps': np.array(timestamps),
        'fps': float(fps),
        'duration': float(duration),
        'total_frames': int(total_frames),
        'video_hash': str(get_video_hash(video_path)),
        'video_name': str(video_path.name),
        'video_path': str(video_path.absolute())
    }


def convert_to_native(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native(v) for v in obj)
    else:
        return obj


def process_all_videos(video_dir):
    extractor = FeatureExtractor()
    exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    videos = [f for f in Path(video_dir).glob('*') if f.suffix.lower() in exts]
    
    if not videos:
        print(f"错误: {video_dir} 中没有视频")
        return None, None
    
    print(f"\n发现 {len(videos)} 个视频")
    
    all_meta, all_feats, offset = [], [], 0
    
    for i, vid in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] ", end="")
        cache = CACHE_DIR / f"{vid.stem}_{get_video_hash(vid)[:8]}.npz"
        
        if cache.exists():
            print(f"加载缓存: {vid.name}")
            try:
                data = np.load(cache, allow_pickle=True)
                vdata = {k: data[k] for k in data.files}
                # 确保所有 numpy 数组转为 list
                vdata['features'] = vdata['features'].tolist() if isinstance(vdata['features'], np.ndarray) else list(vdata['features'])
                vdata['timestamps'] = vdata['timestamps'].tolist() if isinstance(vdata['timestamps'], np.ndarray) else list(vdata['timestamps'])
            except Exception as e:
                print(f"  缓存损坏，重新提取: {e}")
                vdata = extract_video_features(vid, extractor)
                if vdata:
                    np.savez_compressed(cache, **vdata)
        else:
            vdata = extract_video_features(vid, extractor)
            if vdata:
                np.savez_compressed(cache, **vdata)
        
        if not vdata:
            continue
        
        n = len(vdata['features'])
        
        # 构建元数据，确保所有值都是 JSON 可序列化的
        meta_item = {
            'video_name': str(vdata['video_name']),
            'video_path': str(vdata['video_path']),
            'duration': float(vdata['duration']),
            'fps': float(vdata['fps']),
            'n_frames': int(n),
            'start_idx': int(offset),
            'timestamps': [float(t) for t in vdata['timestamps']]  # 确保是 float list
        }
        
        all_meta.append(meta_item)
        all_feats.extend(vdata['features'])
        offset += n
    
    if not all_feats:
        return None, None
    
    all_feats = np.array(all_feats)
    np.save(CACHE_DIR / "features.npy", all_feats)
    
    # 保存前再次确保所有数据可 JSON 序列化
    all_meta_clean = convert_to_native(all_meta)
    
    with open(CACHE_DIR / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(all_meta_clean, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成! 总帧数: {len(all_feats)}, 视频数: {len(all_meta)}")
    return all_feats, all_meta


if __name__ == "__main__":
    process_all_videos(VIDEO_DIR)