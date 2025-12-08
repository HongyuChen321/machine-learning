import os
import patoolib

def batch_decompress():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义源文件夹路径 (原始数据)
    source_dir = os.path.join(current_dir, "原数据集")
    
    # 定义目标根文件夹路径 (dataset)
    target_base_dir = os.path.join(current_dir, "raw")

    # 如果目标根目录不存在，则创建
    if not os.path.exists(target_base_dir):
        os.makedirs(target_base_dir)
        print(f"创建目录: {target_base_dir}")

    # 遍历 1 到 100
    for i in range(1, 101):
        rar_filename = f"{i}.rar"
        rar_path = os.path.join(source_dir, rar_filename)
        
        # 定义该文件对应的解压目标文件夹路径 (例如 dataset/1)
        output_dir = os.path.join(target_base_dir, str(i))

        # 检查源文件是否存在
        if os.path.exists(rar_path):
            print(f"正在处理: {rar_filename} ...")
            
            # 如果目标子文件夹不存在，创建它
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            try:
                # 解压文件到指定目录
                # patoolib 会自动调用系统中的 rar/unrar/7z 等工具
                patoolib.extract_archive(rar_path, outdir=target_base_dir)
                
                print(f"成功解压到: {output_dir}")
            except Exception as e:
                print(f"解压 {rar_filename} 失败: {e}")
        else:
            # 如果文件不存在，打印提示（可选）
            # print(f"跳过: 未找到文件 {rar_filename}")
            pass

if __name__ == "__main__":
    print("开始批量解压缩...")
    batch_decompress()
    print("批量解压缩完成。")