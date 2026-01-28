import h5py
import numpy as np
import tensorflow as tf
import math

class RadioMLSequence(tf.keras.utils.Sequence):
    def __init__(self, hdf5_path, batch_size, indices, num_nodes=32, sigma=1.0, mode='binary'):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.indices = indices
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.mode = mode
        self.num_classes = 2 if mode == 'binary' else 24
        
        # 初始化索引映射
        self.local_indices = np.arange(len(self.indices))
        np.random.shuffle(self.local_indices)
        self.total_len = len(self.indices)

        # --- 计算物理底噪 (只读少量数据) ---
        print("正在计算数据集底噪基准 (仅读取少量样本)...")
        with h5py.File(self.hdf5_path, 'r') as f:
            self.feature_dim = f['X'].shape[1] * f['X'].shape[2] // self.num_nodes
            
            # 只读取前 2000 个样本来估算底噪，而不是读取全部
            sample_size = min(2000, len(self.indices))
            # 注意：这里需要先把 indices 排序才能用于 h5py 读取
            sample_indices = np.sort(self.indices[:sample_size])
            
            temp_Z = f['Z'][sample_indices]
            temp_X = f['X'][sample_indices]
            
            # 找到最小 SNR
            min_snr = np.min(temp_Z)
            noise_idx = np.where(temp_Z == min_snr)[0]
            
            if len(noise_idx) > 0:
                self.noise_std = np.std(temp_X[noise_idx])
            else:
                powers = np.mean(np.var(temp_X, axis=1), axis=1)
                self.noise_std = np.sqrt(np.min(powers))
                
            print(f"✅ 底噪计算完毕: Std={self.noise_std:.6f} (基于 {min_snr}dB 样本)")
            print(f"🚀 数据生成器就绪! (懒加载模式: 训练时实时读取硬盘)")

    def __len__(self):
        return math.ceil(self.total_len / self.batch_size)

    def __getitem__(self, idx):
        # 1. 确定当前 batch 的逻辑索引
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.total_len)
        current_batch_size = end - start
        
        # 获取当前 batch 在原始数据集中的真实索引
        # self.local_indices 是打乱的 0~N，self.indices 是传入的有效样本 ID
        batch_local_idx = self.local_indices[start:end]
        real_indices = self.indices[batch_local_idx]
        
        # h5py 要求索引必须是排序的 (Increasing order)
        # 我们先排序读取，然后再打乱回来 (或者直接使用排序后的数据，对训练影响不大)
        sorted_real_indices = np.sort(real_indices)

        # 2. 实时从硬盘读取数据 (核心修改)
        with h5py.File(self.hdf5_path, 'r') as f:
            X_batch = f['X'][sorted_real_indices]
            Z_batch = f['Z'][sorted_real_indices]
            
            if self.mode != 'binary':
                Y_batch = f['Y'][sorted_real_indices]
            else:
                # 二分类模式下，先给所有样本打上 "H1" (有信号) 标签
                # 后面我们会把一半的数据覆盖为 "H0" (纯噪声)
                # 形状: [batch, 2] -> [H0_prob, H1_prob]
                # 初始化为 [0, 1] 即全部是 H1
                Y_new = np.zeros((current_batch_size, 2), dtype=np.float32)
                Y_new[:, 1] = 1.0 
                Y_batch = Y_new

        # 3. 数据增强/噪声注入 (内存中处理)
        if self.mode == 'binary':
            noise_count = current_batch_size // 2
            if noise_count > 0:
                # 生成纯噪声数据 (H0)
                noise_data = np.random.normal(0, self.noise_std, size=(noise_count, 1024, 2))
                
                # 覆盖后半部分数据
                X_batch[-noise_count:] = noise_data
                Y_batch[-noise_count:, 0] = 1.0 # H0 = 1
                Y_batch[-noise_count:, 1] = 0.0 # H1 = 0
                Z_batch[-noise_count:] = -100   # 标记信噪比极低

        # 4. 转换数据形状以适配 GCN
        X_reshaped = X_batch.reshape(-1, self.num_nodes, self.feature_dim)
        X_tensor = tf.convert_to_tensor(X_reshaped, dtype=tf.float32)
        
        # 5. 动态计算邻接矩阵 A
        # (Batch, Nodes, 1, Feats) - (Batch, 1, Nodes, Feats)
        diff = tf.expand_dims(X_tensor, 2) - tf.expand_dims(X_tensor, 1)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
        A_batch = tf.exp(-dist_sq / (self.sigma ** 2))
        
        # 归一化 A
        D = tf.reduce_sum(A_batch, axis=-1, keepdims=True)
        A_batch_norm = A_batch / (D + 1e-6)

        return [X_tensor, A_batch_norm], Y_batch

    def on_epoch_end(self):
        # 每个 epoch 结束后重新打乱索引，保证随机性
        np.random.shuffle(self.local_indices)

def get_generators(hdf5_path, batch_size=32, num_nodes=32, split_ratio=0.8, max_samples=None):
    # 这一步只读取文件元数据，非常快
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = f['X'].shape[0]
        # 获取特征维度用于后续占位，不读取实际数据
        feature_dim = f['X'].shape[1] * f['X'].shape[2] // num_nodes
        
    if max_samples: total_samples = min(total_samples, max_samples)
    
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    
    split_idx = int(total_samples * split_ratio)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    # 实例化生成器 (现在是轻量级的)
    train_gen = RadioMLSequence(hdf5_path, batch_size, train_indices, num_nodes, mode='binary')
    val_gen = RadioMLSequence(hdf5_path, batch_size, val_indices, num_nodes, mode='binary')
    
    return train_gen, val_gen, 2, feature_dim