import os
import mne
import numpy as np

def is_ecg_channel(name):
    """简单启发式判断通道名是否可能是 ECG（大小写无关）。"""
    n = name.lower()
    keywords = ['ecg', 'ekg', 'ii', 'v5', 'mlii']
    if any(k in n for k in keywords):
        return True
    # 以 X 开头且后跟数字的通道（如 X1..X8）在当前数据集中常为 ECG
    if n.startswith('x') and n[1:].isdigit():
        return True
    return False

def count_ecg_points():
    # 定义文件路径
    # 当前脚本所在目录: .../data
    base_dir = os.path.dirname(__file__)
    # 目标文件夹: .../data/raw/1
    data_folder = os.path.join(base_dir, 'raw', '1')

    rec_filename = "1.rec"
    # MNE库通常需要.edf扩展名，参考main.py进行临时重命名
    edf_filename = "1.edf"
    
    rec_filepath = os.path.join(data_folder, rec_filename)
    edf_filepath = os.path.join(data_folder, edf_filename)
    
    renamed_to_edf = False
    
    try:
        # 1. 临时重命名逻辑 (参考 main.py)
        if os.path.exists(rec_filepath) and not os.path.exists(edf_filepath):
            print(f"正在临时重命名 {rec_filename} -> {edf_filename}")
            os.rename(rec_filepath, edf_filepath)
            renamed_to_edf = True
        elif not os.path.exists(edf_filepath) and not os.path.exists(rec_filepath):
             print(f"[错误] 找不到文件: {rec_filepath}")
             return

        # 2. 使用 MNE 读取数据
        print(f"正在读取 {edf_filepath} ...")
        # preload=True 读取数据到内存
        raw = mne.io.read_raw_edf(edf_filepath, preload=True, verbose='WARNING')
        
        # 3. 统计所有通道的点数，并标注可能的 ECG 通道
        all_channels = raw.ch_names
        print(f"读取成功。文件包含 {len(all_channels)} 个通道。")
        print("正在统计每个通道的点数并标注可能的 ECG 通道:")
        print("-" * 70)
        print(f"{'通道名称':<15} | {'点数':<15} | {'时长(秒)':<10} | 标签")
        print("-" * 70)
        
        ecg_candidates = []
        for ch in all_channels:
            # 获取数据 (mne 返回 data, times)
            data, times = raw[ch]
            point_count = data.shape[1]
            duration = times[-1] if len(times) > 0 else 0
            tag = "[ECG]" if is_ecg_channel(ch) else ""
            if tag:
                ecg_candidates.append((ch, point_count))
            print(f"{ch:<15} | {point_count:<15} | {duration:<10.2f} | {tag}")
            
        print("-" * 70)
        # 汇总输出被判定为 ECG 的通道
        if ecg_candidates:
            print("检测到的 ECG 通道：")
            for ch, cnt in ecg_candidates:
                print(f"  - {ch}: {cnt} 个点")
        else:
            print("未检测到明确的 ECG 通道（可按需要调整判定规则）。")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        
    finally:
        # 4. 恢复文件名 (参考 main.py)
        if renamed_to_edf and os.path.exists(edf_filepath):
            try:
                print(f"正在恢复文件名 {edf_filename} -> {rec_filename}")
                os.rename(edf_filepath, rec_filepath)
            except Exception as e:
                print(f"恢复文件名时出错: {e}")

if __name__ == "__main__":
    count_ecg_points()