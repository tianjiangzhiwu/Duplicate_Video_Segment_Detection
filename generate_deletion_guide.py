#!/usr/bin/env python3
"""
视频重复片段删除指南生成器
分析重复片段，为用户提供删除建议
"""

import json
import cv2
import time
from pathlib import Path
from config import *


def get_video_resolution(video_path):
    """
    获取视频分辨率
    返回 (width, height) 元组
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height
    except Exception as e:
        print(f"无法获取视频 {video_path} 的分辨率: {e}")
        return 0, 0


def calculate_video_quality_score(video_path):
    """
    计算视频质量分数，基于分辨率
    分辨率越高，分数越高
    """
    width, height = get_video_resolution(video_path)
    return width * height  # 简单地使用像素总数作为质量指标


def generate_deletion_guide(duplicates_json_path=None, save_structured_json=True):
    """
    生成视频删除指南

    Args:
        duplicates_json_path: 重复片段JSON文件路径
        save_structured_json: 是否保存结构化的JSON格式指南
    """
    if duplicates_json_path is None:
        duplicates_json_path = OUTPUT_DIR / "duplicates.json"

    if not duplicates_json_path.exists():
        print(f"错误: 未找到重复片段结果文件 {duplicates_json_path}")
        return None

    with open(duplicates_json_path, 'r', encoding='utf-8') as f:
        original_duplicates = json.load(f)

    if not original_duplicates:
        print("没有发现重复片段")
        return None

    print("=" * 80)
    print("视频重复片段删除指南")
    print("=" * 80)

    deletion_guide = []

    # 为了优化用户体验，我们先按视频对进行处理决策，然后按目标视频分组组织删除建议
    # 这样用户可以一次性处理完一个视频中的所有重复片段

    # 首先确定每对视频的保留策略
    pair_decisions = {}
    processed_pairs = set()

    for dup in original_duplicates:
        video_a = dup['video_a_path']
        video_b = dup['video_b_path']

        # 创建排序后的键，保证一致的顺序
        pair_key = tuple(sorted([video_a, video_b]))

        # 如果这对视频还没有处理过，则决定保留哪个视频
        if pair_key not in processed_pairs:
            # 获取两个视频的质量分数
            quality_a = calculate_video_quality_score(video_a)
            quality_b = calculate_video_quality_score(video_b)

            # 根据质量分数决定保留哪个视频
            if quality_a > quality_b:
                # 保留视频A，删除视频B中的重复片段
                video_to_keep = video_a
                video_to_delete_from = video_b
                reason = f"视频A质量更高({quality_a} vs {quality_b})"
            elif quality_b > quality_a:
                # 保留视频B，删除视频A中的重复片段
                video_to_keep = video_b
                video_to_delete_from = video_a
                reason = f"视频B质量更高({quality_b} vs {quality_a})"
            else:
                # 质量分数相同，按字典序保留第一个
                if video_a < video_b:
                    video_to_keep = video_a
                    video_to_delete_from = video_b
                    reason = f"视频质量相同，保留字典序靠前的 {Path(video_a).name}"
                else:
                    video_to_keep = video_b
                    video_to_delete_from = video_a
                    reason = f"视频质量相同，保留字典序靠前的 {Path(video_b).name}"

            print(f"\n处理重复对: {Path(video_a).name} vs {Path(video_b).name}")
            print(f"  → 保留: {Path(video_to_keep).name}")
            print(f"  → 从 {Path(video_to_delete_from).name} 删除相关重复片段")

            # 存储决策
            pair_decisions[pair_key] = {
                'video_to_keep': video_to_keep,
                'video_to_delete_from': video_to_delete_from,
                'reason': reason
            }

            # 标记这对视频已处理
            processed_pairs.add(pair_key)

        # 根据已存储的决策生成删除建议
        decision = pair_decisions[pair_key]
        if decision['video_to_delete_from'] == video_a:
            time_to_delete = dup['time_a']
            time_to_keep = dup['time_b']
        else:
            time_to_delete = dup['time_b']
            time_to_keep = dup['time_a']

        suggestion = {
            'video_to_keep': decision['video_to_keep'],
            'video_to_delete_from': decision['video_to_delete_from'],
            'time_to_delete': time_to_delete,
            'time_to_keep': time_to_keep,
            'duration': dup['duration'],
            'similarity': dup['avg_similarity'],
            'reason': decision['reason']
        }

        print(f"    - {time_to_delete[0]:.2f}s - {time_to_delete[1]:.2f}s "
              f"({dup['duration']:.2f}s, 相似度{dup['avg_similarity']:.3f})")

        deletion_guide.append(suggestion)

    # 按要删除的视频分组，重新排序，让用户可以一次性处理完一个视频
    deletion_by_video = {}
    for suggestion in deletion_guide:
        delete_video = suggestion['video_to_delete_from']
        if delete_video not in deletion_by_video:
            deletion_by_video[delete_video] = []
        deletion_by_video[delete_video].append(suggestion)

    # 为了优化用户体验，按视频名称字母顺序排序，这样用户可以按顺序处理
    sorted_videos = sorted(deletion_by_video.keys(), key=lambda x: Path(x).name)

    print("\n" + "=" * 80)
    print("删除建议摘要（推荐按以下顺序处理）")
    print("=" * 80)

    for video_path in sorted_videos:
        suggestions = deletion_by_video[video_path]
        video_name = Path(video_path).name
        print(f"\n要删除的视频: {video_name}")
        print(f"  路径: {video_path}")
        print("  删除以下时间段的片段:")
        # 按时间顺序排序，便于用户按时间轴处理
        sorted_suggestions = sorted(suggestions, key=lambda x: x['time_to_delete'][0])
        for suggestion in sorted_suggestions:
            time_range = suggestion['time_to_delete']
            reason = suggestion['reason']
            duration = suggestion['duration']
            similarity = suggestion['similarity']
            print(f"    - {time_range[0]:.2f}s - {time_range[1]:.2f}s ({duration:.2f}s, 相似度{similarity:.3f}) - {reason}")

    # 保存文本版删除指南到文件
    guide_text_path = OUTPUT_DIR / "deletion_guide.txt"
    with open(guide_text_path, 'w', encoding='utf-8') as f:
        f.write("视频重复片段删除指南\n")
        f.write("=" * 80 + "\n\n")

        # 按要删除的视频分组写入文本指南（按视频名称排序）
        for video_path in sorted_videos:
            suggestions = deletion_by_video[video_path]
            video_name = Path(video_path).name
            f.write(f"视频: {video_name}\n")
            f.write(f"路径: {video_path}\n")
            f.write("删除以下时间段的片段:\n")
            # 按时间顺序排序
            sorted_suggestions = sorted(suggestions, key=lambda x: x['time_to_delete'][0])
            for suggestion in sorted_suggestions:
                time_range = suggestion['time_to_delete']
                reason = suggestion['reason']
                duration = suggestion['duration']
                similarity = suggestion['similarity']
                f.write(f"  - {time_range[0]:.2f}s - {time_range[1]:.2f}s ({duration:.2f}s, 相似度{similarity:.3f}) - {reason}\n")
            f.write("\n")

    print(f"\n文本版删除指南已保存到: {guide_text_path}")

    # 保存结构化的JSON格式指南，用于自动删除
    if save_structured_json:
        guide_json_path = OUTPUT_DIR / "deletion_guide.json"
        structured_guide = {
            'videos_to_process': [],
            'total_duplicates': len(deletion_guide),
            'guide_created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        for video_path in sorted_videos:
            suggestions = deletion_by_video[video_path]
            video_info = {
                'video_path': video_path,
                'video_name': Path(video_path).name,
                'deletion_segments': []
            }

            sorted_suggestions = sorted(suggestions, key=lambda x: x['time_to_delete'][0])
            for suggestion in sorted_suggestions:
                segment_info = {
                    'time_to_delete': suggestion['time_to_delete'],
                    'duration': suggestion['duration'],
                    'similarity': suggestion['similarity'],
                    'reason': suggestion['reason']
                }
                video_info['deletion_segments'].append(segment_info)

            structured_guide['videos_to_process'].append(video_info)

        with open(guide_json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_guide, f, ensure_ascii=False, indent=2)

        print(f"结构化JSON删除指南已保存到: {guide_json_path}")

    return deletion_guide


def print_usage_example():
    """
    打印使用示例
    """
    print("\n使用示例:")
    print("  1. 先运行主程序检测重复片段:")
    print("     python main.py")
    print("  2. 然后生成删除指南:")
    print("     python generate_deletion_guide.py")


if __name__ == "__main__":
    generate_deletion_guide()