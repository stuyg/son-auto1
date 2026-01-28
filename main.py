import os
import argparse

# ---【新增】强制禁用 cuDNN 自动调优，解决 DNN library initialization failed ---
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
# -----------------------------------------------------------------------

# 1. 先导入 TensorFlow
import tensorflow as tf
import tensorflow as tf

# ==========================================
# 2. 显存配置 (必须在导入 model/dataset 之前执行！)
# ==========================================
# 强制设置环境变量
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ [GPU] 已检测到 {len(gpus)} 个 GPU，显存动态增长已开启。")
    except RuntimeError as e:
        print(f"❌ 显存设置失败: {e}")
else:
    print("⚠️ 未检测到 GPU，将使用 CPU 运行。")

# ==========================================
# 3. 再导入自定义模块 (dataset, model, training)
# ==========================================
# 只有在上面配置完成后，才允许加载这些文件
from dataset import get_generators
from model import GCN_CSS
from training import train_model

def main():
    parser = argparse.ArgumentParser(description="GCN-CSS for RadioML")
    parser.add_argument('--path', type=str, required=True, help='Path to .hdf5 dataset')
    parser.add_argument('--epochs', type=int, default=10)
    # 如果显存依然报错，尝试降低 batch_size 到 16 或 8
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--samples', type=int, default=None)
    
    args = parser.parse_args()
    
    print(f"正在准备数据生成器 (Nodes={args.nodes})...")
    
    # 获取生成器
    train_gen, val_gen, num_classes, num_features = get_generators(
        hdf5_path=args.path,
        batch_size=args.batch_size,
        num_nodes=args.nodes,
        split_ratio=0.8,
        max_samples=args.samples
    )
    
    print(f"生成器准备完毕。分类数: {num_classes}, 节点特征维数: {num_features}")
    
    # 初始化模型
    model = GCN_CSS(num_classes=num_classes, num_nodes=args.nodes)
    
    # Build模型
    model.build([(None, args.nodes, num_features), (None, args.nodes, args.nodes)])
    model.summary()
    
    # 开始训练
    train_model(model, train_gen, val_gen, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()