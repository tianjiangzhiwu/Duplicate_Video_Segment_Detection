import cv2
import json
import base64
from config import *

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>视频重复检测报告</title>
<style>
body{font-family:Microsoft YaHei,Arial;margin:20px;background:#f0f2f5}
h1{color:#333;border-bottom:3px solid #2196F3;padding-bottom:10px}
.summary{background:#fff;padding:20px;border-radius:8px;margin-bottom:20px}
.dup-item{background:#fff;margin-bottom:20px;padding:20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
.dup-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:15px}
.sim-badge{background:#4CAF50;color:#fff;padding:5px 12px;border-radius:20px}
.video-pair{display:flex;gap:20px}
.video-box{flex:1;border:2px solid #e0e0e0;border-radius:8px;padding:15px}
.video-name{font-weight:bold;color:#2196F3;margin-bottom:8px;word-break:break-all}
.time-range{background:#f5f5f5;padding:5px 10px;border-radius:4px;font-family:monospace}
.frames{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap}
.frame{border:1px solid #ddd;border-radius:4px;overflow:hidden}
.frame img{height:100px;display:block}
.frame-label{text-align:center;font-size:12px;color:#999;padding:2px;background:#f9f9f9}
</style>
</head>
<body>
<h1>🎬 视频重复片段检测报告</h1>
<div class="summary">
<h3>📊 统计</h3>
<p>重复片段: <b><!--TOTAL--></b> | 涉及视频: <b><!--VIDS--></b> | 总重复时长: <b><!--DUR-->s</b></p>
</div>
<!--ITEMS-->
</body>
</html>"""

ITEM_TEMPLATE = """
<div class="dup-item">
<div class="dup-header">
<div><b>#<!--IDX--></b> 时长: <!--DUR-->秒</div>
<div class="sim-badge">相似度 <!--SIM--></div>
</div>
<div class="video-pair">
<div class="video-box">
<div class="video-name"><!--VA--></div>
<div class="time-range"><!--TA0-->s - <!--TA1-->s</div>
<div class="frames"><!--FA--></div>
</div>
<div class="video-box">
<div class="video-name"><!--VB--></div>
<div class="time-range"><!--TB0-->s - <!--TB1-->s</div>
<div class="frames"><!--FB--></div>
</div>
</div>
</div>
"""

FRAME_TMPL = '<div class="frame"><img src="data:image/jpeg;base64,<!--IMG-->"><div class="frame-label"><!--T-->s</div></div>'


def get_frame(video, t, size=(160, 120)):
    try:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        frame = cv2.resize(frame, size)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()
    except:
        return None


def gen_frames(video, t0, t1):
    # 在时间段内均匀分布5个时间点
    interval = (t1 - t0) / 4 if t1 > t0 else 0
    times = [t0 + i * interval for i in range(5)]  # 生成5个时间点

    frames_html = []
    for t in times:
        img_data = get_frame(video, t)
        if img_data:
            frames_html.append(FRAME_TMPL.replace("<!--IMG-->", img_data).replace("<!--T-->", str(round(t, 1))))
    return "".join(frames_html)


def generate():
    print("\n生成报告...")
    json_path = OUTPUT_DIR / "duplicates.json"
    if not json_path.exists():
        print("错误: 没有结果文件")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        dups = json.load(f)

    if not dups:
        html = HTML_TEMPLATE.replace("<!--TOTAL-->", "0").replace("<!--VIDS-->", "0").replace("<!--DUR-->", "0")
        html = html.replace("<!--ITEMS-->", "<div style='text-align:center;padding:50px'>无重复片段</div>")
    else:
        # 按视频名称对重复片段进行重新排序，以匹配删除指南的顺序
        # 首先，收集所有需要删除片段的视频
        deletion_candidates = set()
        for dup in dups:
            # 需要确定哪个视频会被标记为"删除"，这需要与删除指南的逻辑一致
            import cv2
            from pathlib import Path

            def get_video_resolution(video_path):
                """获取视频分辨率"""
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    return width * height  # 返回像素总数作为质量评分
                except Exception:
                    return 0

            # 获取两个视频的质量分数
            quality_a = get_video_resolution(dup['video_a_path'])
            quality_b = get_video_resolution(dup['video_b_path'])

            # 根据质量分数决定保留哪个视频（与删除指南相同的逻辑）
            if quality_a > quality_b:
                # 保留视频A，删除视频B中的重复片段
                deletion_candidates.add(dup['video_b_path'])
            elif quality_b > quality_a:
                # 保留视频B，删除视频A中的重复片段
                deletion_candidates.add(dup['video_a_path'])
            else:
                # 质量分数相同，按字典序保留第一个
                if dup['video_a_path'] < dup['video_b_path']:
                    deletion_candidates.add(dup['video_b_path'])
                else:
                    deletion_candidates.add(dup['video_a_path'])

        # 按视频名称排序，这样可以与删除指南的顺序一致
        sorted_deletion_videos = sorted(list(deletion_candidates), key=lambda x: Path(x).name)

        # 按照排序后的视频重新组织重复片段
        reordered_dups = []
        for video_path in sorted_deletion_videos:
            # 找出涉及当前视频的重复片段
            for dup in dups:
                if dup['video_a_path'] == video_path or dup['video_b_path'] == video_path:
                    # 检查是否确实这个视频是要被删除的（根据质量判断）
                    quality_a = get_video_resolution(dup['video_a_path'])
                    quality_b = get_video_resolution(dup['video_b_path'])

                    if quality_a > quality_b:
                        # 保留视频A，删除视频B中的重复片段
                        if dup['video_b_path'] == video_path:
                            reordered_dups.append(dup)
                    elif quality_b > quality_a:
                        # 保留视频B，删除视频A中的重复片段
                        if dup['video_a_path'] == video_path:
                            reordered_dups.append(dup)
                    else:
                        # 质量分数相同，按字典序保留第一个
                        if dup['video_a_path'] < dup['video_b_path']:
                            if dup['video_b_path'] == video_path:
                                reordered_dups.append(dup)
                        else:
                            if dup['video_a_path'] == video_path:
                                reordered_dups.append(dup)

        # 去除重复项，保留唯一项（因为一个视频可能与多个视频重复）
        seen = set()
        unique_reordered_dups = []
        for dup in reordered_dups:
            dup_tuple = (dup['video_a_path'], dup['video_b_path'], tuple(dup['time_a']), tuple(dup['time_b']))
            if dup_tuple not in seen:
                seen.add(dup_tuple)
                unique_reordered_dups.append(dup)

        # 按视频和时间顺序组织显示
        items, all_vids, total_dur = [], set(), 0

        for i, d in enumerate(unique_reordered_dups, 1):
            all_vids.update([d['video_a'], d['video_b']])
            total_dur += d['duration']

            item = ITEM_TEMPLATE.replace("<!--IDX-->", str(i))
            item = item.replace("<!--DUR-->", str(d['duration']))
            item = item.replace("<!--SIM-->", str(d['avg_similarity']))
            item = item.replace("<!--VA-->", d['video_a'])
            item = item.replace("<!--VB-->", d['video_b'])
            item = item.replace("<!--TA0-->", str(d['time_a'][0]))
            item = item.replace("<!--TA1-->", str(d['time_a'][1]))
            item = item.replace("<!--TB0-->", str(d['time_b'][0]))
            item = item.replace("<!--TB1-->", str(d['time_b'][1]))
            item = item.replace("<!--FA-->", gen_frames(d['video_a_path'], d['time_a'][0], d['time_a'][1]))
            item = item.replace("<!--FB-->", gen_frames(d['video_b_path'], d['time_b'][0], d['time_b'][1]))
            items.append(item)

        # 如果还有未包含的重复项（可能是因为质量评估不同导致的），则追加到列表末尾
        existing_items_set = {(d['video_a_path'], d['video_b_path'], tuple(d['time_a']), tuple(d['time_b'])) for d in unique_reordered_dups}
        for d in dups:
            dup_tuple = (d['video_a_path'], d['video_b_path'], tuple(d['time_a']), tuple(d['time_b']))
            if dup_tuple not in existing_items_set:
                all_vids.update([d['video_a'], d['video_b']])
                total_dur += d['duration']

                item = ITEM_TEMPLATE.replace("<!--IDX-->", str(len(items)+1))
                item = item.replace("<!--DUR-->", str(d['duration']))
                item = item.replace("<!--SIM-->", str(d['avg_similarity']))
                item = item.replace("<!--VA-->", d['video_a'])
                item = item.replace("<!--VB-->", d['video_b'])
                item = item.replace("<!--TA0-->", str(d['time_a'][0]))
                item = item.replace("<!--TA1-->", str(d['time_a'][1]))
                item = item.replace("<!--TB0-->", str(d['time_b'][0]))
                item = item.replace("<!--TB1-->", str(d['time_b'][1]))
                item = item.replace("<!--FA-->", gen_frames(d['video_a_path'], d['time_a'][0], d['time_a'][1]))
                item = item.replace("<!--FB-->", gen_frames(d['video_b_path'], d['time_b'][0], d['time_b'][1]))
                items.append(item)

        html = HTML_TEMPLATE.replace("<!--TOTAL-->", str(len(unique_reordered_dups)))
        html = html.replace("<!--VIDS-->", str(len(all_vids)))
        html = html.replace("<!--DUR-->", str(round(total_dur, 1)))
        html = html.replace("<!--ITEMS-->", "".join(items))

    path = OUTPUT_DIR / "report.html"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"报告已保存: {path}")
    return path


if __name__ == "__main__":
    generate()