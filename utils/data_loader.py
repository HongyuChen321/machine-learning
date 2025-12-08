import os
import torch

class DataLoader:
    """五折数据加载器：按折拼接并转换为 epoch 张量。数据位于 ../data/dataset/<n>/"""

    def __init__(self, dataset_root=None, epoch_size=6000, device='cuda'):
        """初始化并立即构建五折的 data_tensor 与 label_tensor（若可用）。"""
        self.epoch_size = epoch_size
        self.device = device if not (device == 'cuda' and not torch.cuda.is_available()) else 'cpu'

        self.data_fold1 = None
        self.data_fold2 = None
        self.data_fold3 = None
        self.data_fold4 = None
        self.data_fold5 = None

        self.label_fold1 = None
        self.label_fold2 = None
        self.label_fold3 = None
        self.label_fold4 = None
        self.label_fold5 = None

        if dataset_root is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            dataset_root = os.path.join(base_dir, 'data', 'dataset')
        self.dataset_root = dataset_root

        for i in range(1, 6):
            data_list = self._concat_files_for_fold(i, dataset_root=self.dataset_root)
            label_list = self._concat_labels_for_fold(i, dataset_root=self.dataset_root)
            data_t, label_t = self._to_epoch_tensors(data_list, label_list,
                                                    epoch_size=self.epoch_size, device=self.device)
            setattr(self, f"data_fold{i}", data_t)
            setattr(self, f"label_fold{i}", label_t)

    def _read_txt(self, path):
        """读取数据文件，每行尝试转为 float，否则保留原字符串，返回列表或 None。"""
        if not os.path.exists(path):
            return None
        out = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s == '':
                    continue
                try:
                    v = float(s)
                except Exception:
                    v = s
                out.append(v)
        return out

    def _read_label_txt(self, path):
        """读取标签文件，每行尝试转为 int，否则保留原字符串，返回列表或 None。"""
        if not os.path.exists(path):
            return None
        out = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s == '':
                    continue
                try:
                    v = int(s)
                except Exception:
                    try:
                        v = int(float(s))
                    except Exception:
                        v = s
                out.append(v)
        return out

    def _concat_files_for_fold(self, fold_idx, dataset_root=None):
        """按折号（1..5）拼接对应编号区间内的 <n>.txt 数据并返回列表。"""
        if not (1 <= fold_idx <= 5):
            return None
        if dataset_root is None:
            dataset_root = self.dataset_root
        start = (fold_idx - 1) * 20 + 1
        end = fold_idx * 20
        result = []
        for n in range(start, end + 1):
            path = os.path.join(dataset_root, str(n), f"{n}.txt")
            vals = self._read_txt(path)
            if vals is None:
                continue
            result.extend(vals)
        return result if result else None

    def _concat_labels_for_fold(self, fold_idx, dataset_root=None):
        """按折号拼接对应编号区间内的 <n>_1.txt 标签并返回列表。"""
        if not (1 <= fold_idx <= 5):
            return None
        if dataset_root is None:
            dataset_root = self.dataset_root
        start = (fold_idx - 1) * 20 + 1
        end = fold_idx * 20
        result = []
        for n in range(start, end + 1):
            path = os.path.join(dataset_root, str(n), f"{n}_1.txt")
            vals = self._read_label_txt(path)
            if vals is None:
                continue
            result.extend(vals)
        return result if result else None

    def _to_epoch_tensors(self, data_list, label_list=None, epoch_size=6000, device='cuda'):
        """把连续数据按 epoch_size 切分并转换为 PyTorch 张量，返回 (data_tensor, label_tensor)。"""
        if data_list is None:
            return None, None
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        data_tensor = torch.tensor(data_list, dtype=torch.float32)
        total_samples = data_tensor.numel()
        n_epochs_from_data = total_samples // epoch_size
        if n_epochs_from_data == 0:
            return None, None

        if label_list is not None:
            n_epochs = min(n_epochs_from_data, len(label_list))
        else:
            n_epochs = n_epochs_from_data

        if n_epochs == 0:
            return None, None

        trim = n_epochs * epoch_size
        data_tensor = data_tensor[:trim].view(n_epochs, epoch_size).to(device)
        if label_list is not None:
            label_tensor = torch.tensor(label_list[:n_epochs], dtype=torch.long, device=device)
        else:
            label_tensor = torch.zeros(n_epochs, dtype=torch.long, device=device)
        return data_tensor, label_tensor
    
if __name__ == "__main__":
    loader = DataLoader()
    print("Fold 1 data shape:", loader.data_fold1.shape if loader.data_fold1 is not None else None)
    print("Fold 1 label shape:", loader.label_fold1.shape if loader.label_fold1 is not None else None)