#!/usr/bin/env python3
"""
自动视频编辑器 - 根据删除指南自动删除重复片段
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import ffmpeg
from config import VIDEO_DIR


def get_video_info(video_path: str) -> Dict:
    """
    获取视频信息包括时长、分辨率等
    """
    try:
        original_path = str(video_path)

        # 首先尝试原路径
        path_obj = Path(original_path)
        if not path_obj.is_absolute():
            path_obj = Path(VIDEO_DIR) / path_obj
            original_path = str(path_obj.resolve())
        else:
            original_path = str(path_obj.resolve())

        # 如果直接路径不存在，尝试在VIDEO_DIR中查找相似名称的文件
        if not Path(original_path).exists():
            # 从路径中提取文件名
            filename = Path(original_path).name
            expected_filename = Path(original_path).name

            # 有时可能因为特殊字符处理导致名称变化，尝试查找相似文件名
            video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']

            # 在VIDEO_DIR中搜索可能的匹配项
            found_file = None
            for ext in video_extensions:
                # 尝试查找包含相同基本名称的文件
                possible_files = list(VIDEO_DIR.glob(f"*{expected_filename.split('.')[0]}*{ext}"))

                for pf in possible_files:
                    # 检查是否是我们要找的文件的变体
                    if expected_filename.split('.')[0] in pf.name or pf.name.replace(' ', '').replace('(', '').replace(')', '') == expected_filename.replace(' ', '').replace('(', '').replace(')', ''):
                        found_file = pf
                        break

                if found_file:
                    break

            if found_file is None:
                # 再尝试一种方法：可能只是文件名格式有差异
                expected_basename = expected_filename.split('.')[0]
                for item in VIDEO_DIR.iterdir():
                    if item.is_file() and item.suffix.lower() in video_extensions:
                        item_basename = item.name.split('.')[0]
                        # 尝试模糊匹配（忽略括号、空格等）
                        clean_expected = expected_basename.replace(' ', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                        clean_item = item_basename.replace(' ', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')

                        if clean_expected == clean_item:
                            found_file = item
                            break

            if found_file is not None:
                print(f"✓ 找到匹配的视频文件: {found_file}")
                original_path = str(found_file.resolve())
            else:
                raise FileNotFoundError(f"无法找到视频文件: {expected_filename} (在 {VIDEO_DIR} 中未找到)")

        print(f"✓ 正在访问视频文件: {original_path}")

        # 使用 ffprobe 命令行工具，它通常比 ffmpeg-python 更可靠
        try:
            # 先尝试使用 ffprobe 命令行工具
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_format', '-show_streams',
                '-print_format', 'json', original_path
            ], capture_output=True, text=True, check=True, timeout=30)

            import json
            probe_result = json.loads(result.stdout)

            # 查找视频流
            video_stream = None
            for stream in probe_result.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if video_stream is None:
                raise ValueError("No video stream found")

            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe_result['format'].get('duration', 0))

            return {
                'width': width,
                'height': height,
                'duration': duration,
                'codec': video_stream.get('codec_name', 'unknown')
            }
        except subprocess.TimeoutExpired:
            print("ffprobe 超时，使用备用方法...")
        except subprocess.CalledProcessError:
            print("ffprobe 执行失败，使用备用方法...")
        except FileNotFoundError:
            print("未找到 ffprobe，尝试使用 OpenCV 获取信息...")

        # 如果 ffprobe 不可用，使用 OpenCV
        cap = cv2.VideoCapture(original_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {original_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            'width': width,
            'height': height,
            'duration': duration,
            'codec': 'unknown'
        }

    except Exception as e:
        print(f"无法获取视频信息 {video_path}: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return {}


def cut_segments_from_video(input_path: str, output_path: str, segments_to_remove: List[Tuple[float, float]],
                           temp_dir: Path = None) -> bool:
    """
    从视频中删除指定的时间段，保留其他部分
    使用 OpenCV 实现视频剪切，不需要 ffmpeg

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        segments_to_remove: 要删除的时间段列表 [(start_time, end_time), ...]
        temp_dir: 临时文件目录

    Returns:
        bool: 是否成功
    """
    if not segments_to_remove:
        # 如果没有要删除的片段，直接复制原文件
        try:
            import shutil
            shutil.copy2(input_path, output_path)
            return True
        except Exception as e:
            print(f"复制视频失败 {input_path}: {e}")
            return False

    # 确保路径格式正确
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    try:
        # 获取视频信息
        video_info = get_video_info(input_path)
        if not video_info:
            print(f"无法获取视频信息: {input_path}")
            return False

        # 计算要保留的时间段
        total_duration = video_info['duration']
        segments_to_remove_sorted = sorted(segments_to_remove)

        # 计算保留的时间段
        keep_segments = []
        current_start = 0.0

        for start, end in segments_to_remove_sorted:
            if current_start < start:
                keep_segments.append((current_start, start))
            current_start = max(current_start, end)

        # 添加最后可能保留的部分
        if current_start < total_duration:
            keep_segments.append((current_start, total_duration))

        # 如果没有保留段，说明整个视频都被标记为删除
        if not keep_segments:
            print(f"警告: 视频 {input_path} 所有部分都标记为删除")
            return False

        # 如果只有一段且包含整个视频，则不需要处理
        if len(keep_segments) == 1 and abs(keep_segments[0][0] - 0.0) < 0.01 and \
           abs(keep_segments[0][1] - total_duration) < 0.01:
            print(f"视频 {input_path} 没有需要删除的片段")
            return False

        print(f"视频 {input_path} 需要删除 {len(segments_to_remove)} 个片段，保留 {len(keep_segments)} 个片段")
        print("正在使用 OpenCV 进行视频剪切...")

        # 使用 OpenCV 读取原视频并写入保留的片段
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {input_path}")
            return False

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_time = 0  # 当前时间（秒）

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算当前时间
            current_time = frame_count / fps

            # 检查当前时间是否在保留的时间段内
            keep_frame = False
            for start_time, end_time in keep_segments:
                if start_time <= current_time <= end_time:
                    keep_frame = True
                    break

            # 如果当前时间在保留段内，写入帧
            if keep_frame:
                out.write(frame)

            frame_count += 1

            # 每处理10%进度打印一次状态
            if frame_count % max(1, total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  进度: {progress:.1f}%")

        # 释放资源
        cap.release()
        out.release()

        print(f"视频处理完成: {output_path}")
        return True

    except Exception as e:
        print(f"处理视频失败 {input_path}: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return False


def execute_auto_deletion(duplicates_json_path=None, deletion_guide_json_path=None):
    """
    执行自动删除功能

    Args:
        duplicates_json_path: 重复片段JSON文件路径
        deletion_guide_json_path: 删除指南JSON文件路径
    """
    if deletion_guide_json_path is None:
        deletion_guide_json_path = Path("output") / "deletion_guide.json"

    # 优先使用结构化的JSON指南文件
    if deletion_guide_json_path.exists():
        print("使用结构化的删除指南文件...")

        with open(deletion_guide_json_path, 'r', encoding='utf-8') as f:
            structured_guide = json.load(f)

        sorted_videos_data = structured_guide.get('videos_to_process', [])

        if not sorted_videos_data:
            print("没有找到可处理的视频")
            return True

        # 验证视频文件是否存在，如果原始路径不存在则尝试在当前视频目录下查找
        existing_videos_data = []
        for video_data in sorted_videos_data:
            video_path = video_data['video_path']
            video_path_obj = Path(video_path)

            # 如果视频在原始路径存在，直接使用
            if video_path_obj.exists():
                existing_videos_data.append(video_data)
            else:
                # 尝试在视频目录中查找同名文件
                video_filename = video_path_obj.name
                potential_paths = [
                    VIDEO_DIR / video_filename,
                    Path(str(video_path_obj).replace('\\\\', '\\')),  # 尝试修复路径格式
                    Path(str(video_path_obj).replace('\\', '/')),     # 统一使用正斜杠
                    video_path_obj                                    # 原始路径
                ]

                found = False
                for pot_path in potential_paths:
                    if pot_path.exists():
                        # 更新视频路径
                        updated_video_data = video_data.copy()
                        updated_video_data['video_path'] = str(pot_path)
                        existing_videos_data.append(updated_video_data)
                        found = True
                        break

                if not found:
                    print(f"跳过不存在的视频: {video_path} (在 {VIDEO_DIR} 中也未找到 {video_filename})")

        if not existing_videos_data:
            print("没有找到任何存在的视频文件，无法执行删除操作")
            return False

        print(f"\n将自动处理 {len(existing_videos_data)} 个视频中的重复片段:")
        for video_data in existing_videos_data:
            video_path = video_data['video_path']
            video_name = video_data['video_name']
            segments_count = len(video_data['deletion_segments'])
            print(f"  - {video_name} (需删除 {segments_count} 个片段)")

        # 用户确认
        response = input(f"\n确认开始自动删除这 {len(existing_videos_data)} 个视频中的重复片段吗？[Y/N]: ").strip().lower()
        if response not in ('y', 'yes', '是'):
            print("用户取消操作")
            return False

        # 处理每个视频
        success_count = 0
        total_count = len(existing_videos_data)

        for i, video_data in enumerate(existing_videos_data, 1):
            video_path = video_data['video_path']

            if not Path(video_path).exists():
                print(f"[{i}/{total_count}] 跳过不存在的视频: {video_path}")
                continue

            video_name = Path(video_path).name
            segments_to_remove = [seg['time_to_delete'] for seg in video_data['deletion_segments']]

            print(f"\n[{i}/{total_count}] 处理视频: {video_name}")
            print(f"  需删除 {len(segments_to_remove)} 个片段:")
            for j, time_range in enumerate(segments_to_remove):
                print(f"    {j+1}. {time_range[0]:.2f}s - {time_range[1]:.2f}s")

            # 创建输出路径（原文件名 + _cleaned）
            video_path_obj = Path(video_path)
            output_path = video_path_obj.parent / f"{video_path_obj.stem}_cleaned{video_path_obj.suffix}"

            # 执行删除操作
            if cut_segments_from_video(video_path, output_path, segments_to_remove):
                print(f"  ✓ 处理成功，临时输出到: {output_path}")

                # 询问是否替换原文件
                replace_response = input(f"  是否用处理后的视频替换原视频 {video_path}? [Y/N]: ").strip().lower()
                if replace_response in ('y', 'yes', '是'):
                    try:
                        # 删除原文件
                        Path(video_path).unlink()

                        # 将处理后的文件移动到原位置
                        output_path.rename(video_path)

                        print(f"  ✓ 已成功替换原视频: {video_path}")
                        success_count += 1
                    except Exception as e:
                        print(f"  ✗ 替换原文件失败: {e}")
                        print(f"  临时文件仍存在于: {output_path}")
                else:
                    print(f"  原文件未替换，临时文件保存在: {output_path}")
            else:
                print(f"  ✗ 处理失败: {video_path}")

        print(f"\n完成！成功处理 {success_count}/{total_count} 个视频")
        return success_count == total_count

    # 如果没有结构化JSON文件，则回退到原来的逻辑
    else:
        if duplicates_json_path is None:
            duplicates_json_path = Path("output") / "duplicates.json"

        if not duplicates_json_path.exists():
            print(f"错误: 未找到重复片段结果文件 {duplicates_json_path}")
            return False

        # 读取重复片段数据
        with open(duplicates_json_path, 'r', encoding='utf-8') as f:
            original_duplicates = json.load(f)

        if not original_duplicates:
            print("没有发现重复片段，无需删除")
            return True

        # 生成结构化的删除指南（类似generate_deletion_guide的功能）
        print("正在分析删除指南...")
        pair_decisions = {}
        processed_pairs = set()
        deletion_guide = []

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

            deletion_guide.append(suggestion)

        # 按要删除的视频分组
        deletion_by_video = {}
        for suggestion in deletion_guide:
            delete_video = suggestion['video_to_delete_from']
            if delete_video not in deletion_by_video:
                deletion_by_video[delete_video] = []
            deletion_by_video[delete_video].append(suggestion)

        # 检查视频文件是否存在
        existing_videos = []
        for video_path in deletion_by_video.keys():
            video_path_obj = Path(video_path)

            # 如果视频在原始路径存在，直接使用
            if video_path_obj.exists():
                existing_videos.append(video_path)
            else:
                # 尝试在视频目录中查找同名文件
                video_filename = video_path_obj.name
                potential_paths = [
                    VIDEO_DIR / video_filename,
                    Path(str(video_path_obj).replace('\\\\', '\\')),  # 尝试修复路径格式
                    Path(str(video_path_obj).replace('\\', '/')),     # 统一使用正斜杠
                    video_path_obj                                    # 原始路径
                ]

                for pot_path in potential_paths:
                    if pot_path.exists():
                        existing_videos.append(str(pot_path))
                        break
                else:
                    print(f"跳过不存在的视频: {video_path}")

        if not existing_videos:
            print("没有找到任何存在的视频文件，无法执行删除操作")
            return False

        print(f"\n将自动处理 {len(existing_videos)} 个视频中的重复片段:")
        for video_path in existing_videos:
            # 找到对应的删除建议
            matched_key = next((key for key in deletion_by_video.keys()
                              if Path(key).name == Path(video_path).name),
                             list(deletion_by_video.keys())[0])
            suggestions = deletion_by_video[matched_key]

            video_name = Path(video_path).name
            print(f"  - {video_name} (需删除 {len(suggestions)} 个片段)")

        # 用户确认
        response = input(f"\n确认开始自动删除这 {len(existing_videos)} 个视频中的重复片段吗？[Y/N]: ").strip().lower()
        if response not in ('y', 'yes', '是'):
            print("用户取消操作")
            return False

        # 处理每个视频
        success_count = 0
        total_count = len(existing_videos)

        for i, video_path in enumerate(existing_videos, 1):
            # 找到对应的删除建议
            matched_key = next((key for key in deletion_by_video.keys()
                              if Path(key).name == Path(video_path).name),
                             list(deletion_by_video.keys())[0])
            suggestions = deletion_by_video[matched_key]

            if not Path(video_path).exists():
                print(f"[{i}/{total_count}] 跳过不存在的视频: {video_path}")
                continue

            video_name = Path(video_path).name
            print(f"\n[{i}/{total_count}] 处理视频: {video_name}")
            print(f"  需删除 {len(suggestions)} 个片段:")
            for j, suggestion in enumerate(suggestions):
                time_range = suggestion['time_to_delete']
                duration = suggestion['duration']
                similarity = suggestion['similarity']
                print(f"    {j+1}. {time_range[0]:.2f}s - {time_range[1]:.2f}s ({duration:.2f}s, 相似度{similarity:.3f})")

            # 准备删除时间段列表
            segments_to_remove = [suggestion['time_to_delete'] for suggestion in suggestions]

            # 创建输出路径（原文件名 + _cleaned）
            video_path_obj = Path(video_path)
            output_path = video_path_obj.parent / f"{video_path_obj.stem}_cleaned{video_path_obj.suffix}"

            # 执行删除操作
            if cut_segments_from_video(video_path, output_path, segments_to_remove):
                print(f"  ✓ 处理成功，临时输出到: {output_path}")

                # 询问是否替换原文件
                replace_response = input(f"  是否用处理后的视频替换原视频 {video_path}? [Y/N]: ").strip().lower()
                if replace_response in ('y', 'yes', '是'):
                    try:
                        # 删除原文件
                        Path(video_path).unlink()

                        # 将处理后的文件移动到原位置
                        output_path.rename(video_path)

                        print(f"  ✓ 已成功替换原视频: {video_path}")
                        success_count += 1
                    except Exception as e:
                        print(f"  ✗ 替换原文件失败: {e}")
                        print(f"  临时文件仍存在于: {output_path}")
                else:
                    print(f"  原文件未替换，临时文件保存在: {output_path}")
            else:
                print(f"  ✗ 处理失败: {video_path}")

        print(f"\n完成！成功处理 {success_count}/{total_count} 个视频")
        return success_count == total_count


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


if __name__ == "__main__":
    # 仅供测试用
    print("自动视频编辑器模块")
    print("此模块应通过main.py调用execute_auto_deletion()函数来执行自动删除功能")