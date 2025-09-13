import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CSVDataProcessor:
    def __init__(self, csv_path, save_path, sequence_length=20):
        self.csv_path = csv_path
        self.save_path = save_path
        self.sequence_length = sequence_length
        
    def load_and_clean_data(self):
        """加载并清理CSV数据"""
        print("Loading CSV data...")
        df = pd.read_csv(self.csv_path)
        
        # 转换时间戳
        df['datetime'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 基本数据清理
        df = df.dropna()
        df = df[df['volume'] >= 0]  # 移除异常交易量
        df = df[df['high'] >= df['low']]  # 确保high >= low
        df = df[df['close'] > 0]  # 确保价格为正
        
        print(f"Data loaded: {len(df)} records")
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        return df
    
    def prepare_features(self, df):
        """准备基础特征（保持原始数据）"""
        print("Preparing basic features...")
        
        # 只使用基础的OHLCV特征
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # 直接使用原始数据，不进行标准化
        # 让qlib框架自己处理数据标准化
        feature_data = df[feature_cols].values
        
        return feature_data, feature_cols
    
    def split_data_by_time(self, df, feature_data):
        """按时间划分数据集"""
        print("Splitting data by time...")
        
        n = len(df)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        data_splits = {
            'train': {
                'feature': feature_data[:train_end],
                'datetime': df['datetime'].iloc[:train_end].tolist(),
                'close': df['close'].iloc[:train_end].values
            },
            'valid': {
                'feature': feature_data[train_end:val_end],
                'datetime': df['datetime'].iloc[train_end:val_end].tolist(),
                'close': df['close'].iloc[train_end:val_end].values
            },
            'test': {
                'feature': feature_data[val_end:],
                'datetime': df['datetime'].iloc[val_end:].tolist(),
                'close': df['close'].iloc[val_end:].values
            }
        }
        
        for split_name, split_data in data_splits.items():
            print(f"{split_name}: {len(split_data['feature'])} samples")
        
        return data_splits
    
    def process_and_save(self):
        """完整的数据处理流程"""
        print("Starting data processing...")
        
        # 1. 加载数据
        df = self.load_and_clean_data()
        
        # 2. 准备基础特征（原始OHLCV数据）
        feature_data, feature_cols = self.prepare_features(df)
        
        # 3. 按时间划分数据集
        data_splits = self.split_data_by_time(df, feature_data)
        
        # 4. 保存数据
        os.makedirs(self.save_path, exist_ok=True)
        
        # 保存每个数据集
        for split_name, split_data in data_splits.items():
            save_file = os.path.join(self.save_path, f'{split_name}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {split_name} data to {save_file}")
        
        # 5. 保存元信息
        meta_info = {
            'feature_dim': len(feature_cols),
            'feature_cols': feature_cols,
            'sequence_length': self.sequence_length,
            'data_stats': {
                'total_samples': len(feature_data),
                'train_samples': len(data_splits['train']['feature']),
                'valid_samples': len(data_splits['valid']['feature']),
                'test_samples': len(data_splits['test']['feature'])
            }
        }
        
        with open(os.path.join(self.save_path, 'meta_info.pkl'), 'wb') as f:
            pickle.dump(meta_info, f)
        
        print("Data processing completed!")
        print(f"Feature dimension: {meta_info['feature_dim']}")
        print(f"Features: {feature_cols}")
        print(f"Total samples: {meta_info['data_stats']['total_samples']}")
        
        return meta_info

if __name__ == "__main__":
    # 使用示例
    processor = CSVDataProcessor(
        csv_path="./data/XAUUSDM1_utf8.csv",  # 替换为您的CSV文件路径
        save_path="./data/processed_datasets",
        sequence_length=20
    )
    
    meta_info = processor.process_and_save()
