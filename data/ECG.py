import os
import mne
import numpy as np
import shutil

def process_folder(folder_path, folder_name, base_dir):
    rec_name = f"{folder_name}.rec"
    edf_name = f"{folder_name}.edf"
    rec_path = os.path.join(folder_path, rec_name)
    edf_path = os.path.join(folder_path, edf_name)

    out_folder = os.path.join(base_dir, 'dataset_raw', folder_name)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{folder_name}.txt")

    renamed = False
    try:
        # 如果有 .rec 且没有 .edf，则临时重命名为 .edf 以便 mne 读取
        if os.path.exists(rec_path) and not os.path.exists(edf_path):
            os.rename(rec_path, edf_path)
            renamed = True
        # 如果既无 .rec 也无 .edf，跳过
        elif not os.path.exists(edf_path) and not os.path.exists(rec_path):
            return None

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')

        # 查找以 x2 开头或名为 25 的通道（不区分大小写），使用第一个找到的通道
        x2_chs = [ch for ch in raw.ch_names if ch.lower().startswith('x2') or ch.lower() == '25']
        if not x2_chs:
            # 即便没有 x2/25，也复制指定的两个辅助文件到 dataset/<i>
            for fname in (f"{folder_name}_1.txt", f"{folder_name}_1.xlsx"):
                src = os.path.join(folder_path, fname)
                dst = os.path.join(out_folder, fname)
                if os.path.exists(src):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
            # 返回该文件夹名和通道列表，供主流程汇总
            return {'name': folder_name, 'channels': list(raw.ch_names)}

        ch = x2_chs[0]
        if len(x2_chs) > 1:
            # 提示已选择的通道（无需详细输出，可改为日志）
            print(f"[提示] 在 {folder_name} 检测到多个候选通道，已使用第一个: {ch}")

        data, _ = raw.copy().pick_channels([ch])[:]  # shape (1, n)
        data = data.flatten()

        # 仅保存数据值，每行一个，使用 repr 保留原始表示
        with open(out_path, 'w', encoding='utf-8') as f:
            for v in data:
                f.write(repr(float(v)) + '\n')

        # 复制指定的两个辅助文件：<n>_1.txt 和 <n>_1.xlsx（若存在）
        for fname in (f"{folder_name}_1.txt", f"{folder_name}_1.xlsx"):
            src = os.path.join(folder_path, fname)
            dst = os.path.join(out_folder, fname)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

        return None

    except Exception as e:
        print(f"处理 {folder_name} 时出错: {e}")
        return None
    finally:
        # 恢复文件名
        if renamed and os.path.exists(edf_path):
            try:
                os.rename(edf_path, rec_path)
            except Exception:
                pass

def main():
    base_dir = os.path.dirname(__file__)
    raw_root = os.path.join(base_dir, 'raw')
    if not os.path.isdir(raw_root):
        print(f"[错误] 未找到 raw 根目录: {raw_root}")
        return

    missing = []
    # 仅遍历名字为数字的子文件夹（例如 1..100）
    for name in sorted(os.listdir(raw_root), key=lambda x: int(x) if x.isdigit() else x):
        if not name.isdigit():
            continue
        folder_path = os.path.join(raw_root, name)
        if os.path.isdir(folder_path):
            # 简短提示
            print(f"处理 {name}")
            res = process_folder(folder_path, name, base_dir)
            if res:
                missing.append(res)

    # 最后汇总并打印缺少 x2/25 的文件夹信息
    if missing:
        print("\n以下文件夹未检测到以 x2 开头或名为 25 的通道：")
        for item in missing:
            chs = ", ".join(item['channels'])
            print(f"{item['name']}: {chs}")
    else:
        print("\n所有处理的文件夹均检测到 x2 或 25 通道。")

if __name__ == "__main__":
    main()