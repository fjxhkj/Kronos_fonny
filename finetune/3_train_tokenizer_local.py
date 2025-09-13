import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import pickle
from tqdm import tqdm
import argparse

class FinancialDataset(Dataset):
    def __init__(self, data_path):
        # 加载pkl文件
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 根据您的预处理脚本，数据结构是字典，包含 'feature', 'datetime', 'close'
        if not isinstance(self.data, dict):
            raise ValueError(f"Expected dict data format, got {type(self.data)}")
        
        if 'feature' not in self.data:
            raise ValueError(f"'feature' key not found in data. Available keys: {list(self.data.keys())}")
        
        # 提取特征数据
        self.features = self.data['feature']
        
        # 确保是numpy数组
        if not isinstance(self.features, np.ndarray):
            self.features = np.array(self.features)
        
        print(f"Loaded data shape: {self.features.shape}")
        print(f"Data type: {self.features.dtype}")
        print(f"Feature dimension: {self.features.shape[1] if len(self.features.shape) > 1 else 1}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)

class KronosTokenizer(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # 编码器 - 按照官方架构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 码本 - 官方使用可学习的embedding
        self.codebook = nn.Embedding(vocab_size, input_dim)
        
        # 初始化码本
        nn.init.uniform_(self.codebook.weight, -1/vocab_size, 1/vocab_size)
    
    def encode(self, x):
        # 获取logits
        logits = self.encoder(x)
        
        # 使用Gumbel softmax进行可微分的离散化
        if self.training:
            # 训练时使用Gumbel softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            soft_tokens = torch.softmax((logits + gumbel_noise) / 1.0, dim=-1)
            
            # 直通估计器
            hard_tokens = torch.argmax(soft_tokens, dim=-1)
            tokens = soft_tokens + (torch.nn.functional.one_hot(hard_tokens, self.vocab_size).float() - soft_tokens).detach()
        else:
            # 推理时直接使用argmax
            hard_tokens = torch.argmax(logits, dim=-1)
            tokens = torch.nn.functional.one_hot(hard_tokens, self.vocab_size).float()
        
        return tokens, hard_tokens
    
    def decode(self, tokens):
        if tokens.dim() == 1:
            # 如果是token indices
            return self.codebook(tokens)
        else:
            # 如果是one-hot向量
            return torch.matmul(tokens, self.codebook.weight)
    
    def forward(self, x):
        tokens, token_ids = self.encode(x)
        reconstructed = self.decode(tokens)
        return reconstructed, token_ids

def load_meta_info(data_dir):
    """加载元信息"""
    meta_path = os.path.join(data_dir, 'meta_info.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta_info = pickle.load(f)
        print("Meta information loaded:")
        print(f"  Feature dimension: {meta_info['feature_dim']}")
        print(f"  Feature columns: {meta_info['feature_cols']}")
        print(f"  Data stats: {meta_info['data_stats']}")
        return meta_info
    else:
        print("No meta_info.pkl found")
        return None

def train_tokenizer(args):
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 固定数据路径
    data_dir = './data/processed_datasets'
    train_path = os.path.join(data_dir, 'train.pkl')
    
    # 检查文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    # 加载元信息
    meta_info = load_meta_info(data_dir)
    
    # 加载训练数据集
    train_dataset = FinancialDataset(train_path)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # 加载验证数据集
    valid_path = os.path.join(data_dir, 'valid.pkl')
    valid_dataloader = None
    if os.path.exists(valid_path):
        valid_dataset = FinancialDataset(valid_path)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        print(f"Validation dataset loaded: {len(valid_dataset)} samples")
    
    # 获取数据维度
    sample_data = train_dataset[0]
    input_dim = sample_data.shape[0] if sample_data.dim() > 0 else 1
    print(f"Input dimension: {input_dim}")
    print(f"Training dataset size: {len(train_dataset)}")
    
    # 验证维度与meta_info一致性
    if meta_info and input_dim != meta_info['feature_dim']:
        print(f"Warning: Input dim {input_dim} != meta feature dim {meta_info['feature_dim']}")
    
    # 初始化模型
    model = KronosTokenizer(
        input_dim=input_dim,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    print(f"Model initialized with:")
    print(f"  Input dim: {input_dim}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Hidden dim: {args.hidden_dim}")
    
    # 优化器 - 按照官方设置
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 损失函数
    mse_loss = nn.MSELoss()
    
    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        total_mse = 0
        
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, batch_data in enumerate(pbar):
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            reconstructed, token_ids = model(batch_data)
            
            # 重构损失
            recon_loss = mse_loss(reconstructed, batch_data)
            
            # 总损失
            loss = recon_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += recon_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'MSE': f'{recon_loss.item():.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_dataloader)
        avg_mse = total_mse / len(train_dataloader)
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Average Loss: {avg_loss:.6f}')
        print(f'  Average MSE: {avg_mse:.6f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 验证（如果有验证集）
        if valid_dataloader is not None:
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_data in valid_dataloader:
                    batch_data = batch_data.to(device)
                    reconstructed, _ = model(batch_data)
                    valid_loss += mse_loss(reconstructed, batch_data).item()
            
            valid_loss /= len(valid_dataloader)
            print(f'  Validation MSE: {valid_loss:.6f}')
            model.train()
        
        # 保存检查点
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args,
                'meta_info': meta_info
            }
            torch.save(checkpoint, f'{args.save_dir}/checkpoint_epoch_{epoch+1}.pt')
    
    # 保存最终模型
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': args,
        'meta_info': meta_info
    }
    torch.save(final_checkpoint, f'{args.save_dir}/final_tokenizer.pt')
    
    print("Training completed!")
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Kronos Tokenizer')
    
    # 模型相关参数 - 按照官方推荐
    parser.add_argument('--vocab_size', type=int, default=65536,
                       help='Vocabulary size (official recommends 2^16 or 2^18)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # 保存相关参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 开始训练
    model = train_tokenizer(args)

if __name__ == '__main__':
    main()
