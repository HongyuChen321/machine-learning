import os

FS = 200
EPOCH_SECONDS = 30
EPOCHS_TO_REMOVE = 30
SAMPLES_TO_REMOVE = FS * EPOCH_SECONDS * EPOCHS_TO_REMOVE  # 200 * 30 * 30 = 180000

def process_one(folder_path, folder_name, base_dir):
    src_folder = os.path.join(base_dir, 'dataset_raw', folder_name)
    if not os.path.isdir(src_folder):
        return f"missing_folder:{folder_name}"

    out_folder = os.path.join(base_dir, 'dataset', folder_name)
    os.makedirs(out_folder, exist_ok=True)

    data_txt = os.path.join(src_folder, f"{folder_name}.txt")
    label_txt = os.path.join(src_folder, f"{folder_name}_1.txt")

    # 数据文件（每行一个样本）
    if not os.path.exists(data_txt):
        return f"no_data:{folder_name}"
    with open(data_txt, 'r', encoding='utf-8') as f:
        sample_lines = f.readlines()
    if len(sample_lines) <= SAMPLES_TO_REMOVE:
        return f"not_enough_samples:{folder_name}"
    kept_samples = sample_lines[: len(sample_lines) - SAMPLES_TO_REMOVE]
    out_data_txt = os.path.join(out_folder, f"{folder_name}.txt")
    with open(out_data_txt, 'w', encoding='utf-8') as f:
        f.writelines(kept_samples)

    # label txt（每行代表一个 epoch），去掉最后 EPOCHS_TO_REMOVE 行
    if os.path.exists(label_txt):
        with open(label_txt, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
        kept_label_lines = label_lines[: max(0, len(label_lines) - EPOCHS_TO_REMOVE)]
        out_label_txt = os.path.join(out_folder, f"{folder_name}_1.txt")
        with open(out_label_txt, 'w', encoding='utf-8') as f:
            f.writelines(kept_label_lines)

    return None  # 成功

def main():
    base_dir = os.path.dirname(__file__)
    raw_dataset_root = os.path.join(base_dir, 'dataset_raw')
    if not os.path.isdir(raw_dataset_root):
        print(f"[错误] 未找到 dataset_raw 目录: {raw_dataset_root}")
        return

    problems = []
    names = sorted(os.listdir(raw_dataset_root), key=lambda x: int(x) if x.isdigit() else x)
    for name in names:
        if not name.isdigit():
            continue
        folder_path = os.path.join(raw_dataset_root, name)
        if not os.path.isdir(folder_path):
            continue
        print(f"处理 {name}")
        res = process_one(folder_path, name, base_dir)
        if res is not None:
            problems.append(res)

    if problems:
        print("\n以下问题：")
        for p in problems:
            print(p)
    else:
        print("\n处理完成，所有 txt 文件已保存到 dataset/<n>/ 下。")

if __name__ == "__main__":
    main()