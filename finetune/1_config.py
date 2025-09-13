import torch

# 数据配置
csv_data_path = "./data/XAUUSDM1_utf8.csv"  # 您的CSV文件路径
dataset_path = "./data/processed_datasets"         # 处理后数据保存路径
save_path = "./checkpoints"               # 模型检查点保存路径

# 预训练模型路径
pretrained_tokenizer_path = "./ckpt/tokenizer"
pretrained_predictor_path = "./ckpt/predictor"

# 训练超参数
tokenizer_config = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'gradient_clip_val': 1.0
}

predictor_config = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'num_epochs': 30,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_clip_val': 1.0
}

# 设备自适应配置
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

device = get_device()

# 模型配置
model_config = {
    'sequence_length': 20,
    'feature_dim': 14,  # 根据特征数量调整
    'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
    'dropout': 0.1
}
