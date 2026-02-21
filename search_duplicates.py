import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from config import *
from build_index import load_index

def find_duplicates():
    print("\n" + "="*50)
    print("检索重复片段")
    print("="*50)
    
    features = np.load(CACHE_DIR / "features.npy")
    
    # 修复：指定 UTF-8 编码读取
    with open(CACHE_DIR / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    index = load_index()
    frame_to_video = []
    for vid_idx, meta in enumerate(metadata):
        frame_to_video.extend([vid_idx] * meta['n_frames'])
    
    print(f"总帧数: {len(features)}, 阈值: {SIMILARITY_THRESHOLD}")
    
    matches = []
    batch = 1000
    
    for start in tqdm(range(0, len(features), batch), desc="检索", ncols=70):
        end = min(start + batch, len(features))
        dists, idxs = index.search(features[start:end], TOP_K + 1)
        
        for i, (d, idx) in enumerate(zip(dists, idxs)):
            qf = start + i
            qv = frame_to_video[qf]
            for dist, mf in zip(d[1:], idx[1:]):  # 跳过自身
                if dist < SIMILARITY_THRESHOLD:
                    continue
                mv = frame_to_video[mf]
                if qv == mv or qf > mf:
                    continue
                
                qm, mm = metadata[qv], metadata[mv]
                matches.append({
                    'qv': qv, 'mv': mv,
                    'qt': qm['timestamps'][qf - qm['start_idx']],
                    'mt': mm['timestamps'][mf - mm['start_idx']],
                    'sim': float(dist)
                })
    
    print(f"找到 {len(matches)} 对相似帧，聚合成片段...")
    
    # 按视频对分组聚合
    pair_matches = defaultdict(list)
    for m in matches:
        key = (m['qv'], m['mv'])
        pair_matches[key].append(m)
    
    results = []
    for (va, vb), ms in pair_matches.items():
        ms.sort(key=lambda x: x['qt'])
        segs = aggregate(ms)
        for s in segs:
            results.append({
                'video_a': metadata[va]['video_name'],
                'video_b': metadata[vb]['video_name'],
                'time_a': [round(s['sa'], 2), round(s['ea'], 2)],
                'time_b': [round(s['sb'], 2), round(s['eb'], 2)],
                'duration': round(s['dur'], 2),
                'avg_similarity': round(s['sim'], 3),
                'video_a_path': metadata[va]['video_path'],
                'video_b_path': metadata[vb]['video_path']
            })
    
    results.sort(key=lambda x: x['duration'], reverse=True)
    print(f"发现 {len(results)} 个重复片段")
    return results


def aggregate(matches):
    if not matches:
        return []
    
    segs, cur = [], {
        'sa': matches[0]['qt'], 'ea': matches[0]['qt'],
        'sb': matches[0]['mt'], 'eb': matches[0]['mt'],
        'sims': [matches[0]['sim']]
    }
    max_gap = SAMPLE_INTERVAL * 2.5
    
    for m in matches[1:]:
        ga, gb = m['qt'] - cur['ea'], m['mt'] - cur['eb']
        if ga <= max_gap and abs(ga - gb) < max_gap:
            cur['ea'], cur['eb'] = m['qt'], m['mt']
            cur['sims'].append(m['sim'])
        else:
            if cur['ea'] - cur['sa'] >= MIN_DUPLICATE_DURATION:
                segs.append({
                    'sa': cur['sa'], 'ea': cur['ea'],
                    'sb': cur['sb'], 'eb': cur['eb'],
                    'dur': cur['ea'] - cur['sa'],
                    'sim': sum(cur['sims']) / len(cur['sims'])
                })
            cur = {'sa': m['qt'], 'ea': m['qt'], 'sb': m['mt'], 'eb': m['mt'], 'sims': [m['sim']]}
    
    if cur['ea'] - cur['sa'] >= MIN_DUPLICATE_DURATION:
        segs.append({
            'sa': cur['sa'], 'ea': cur['ea'],
            'sb': cur['sb'], 'eb': cur['eb'],
            'dur': cur['ea'] - cur['sa'],
            'sim': sum(cur['sims']) / len(cur['sims'])
        })
    
    return segs


if __name__ == "__main__":
    dups = find_duplicates()
    
    # 修复：指定 UTF-8 编码写入
    with open(OUTPUT_DIR / "duplicates.json", 'w', encoding='utf-8') as f:
        json.dump(dups, f, ensure_ascii=False, indent=2)