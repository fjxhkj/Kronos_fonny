# =======================
# 文件名: future_prediction_modified.py
# 基于官方 prediction 示例修改，实现真正的未来预测
# =======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def generate_future_timestamps(last_timestamp, pred_len, frequency='H1'):
    """
    生成未来时间戳 - 这是关键的修改点

    参数说明:
    last_timestamp: pandas.Timestamp, 最后一个历史时间戳
    pred_len: int, 要预测的周期数
    frequency: str, 数据频率

    返回:
    pandas.Series, 未来时间戳序列
    """

    # 时间频率映射
    freq_mapping = {
        'M1': timedelta(minutes=1),
        'M5': timedelta(minutes=5),
        'M15': timedelta(minutes=15),
        'M30': timedelta(minutes=30),
        'H1': timedelta(hours=1),
        'H4': timedelta(hours=4),
        'D1': timedelta(days=1)
    }

    if frequency not in freq_mapping:
        raise ValueError(f"不支持的频率: {frequency}")

    time_delta = freq_mapping[frequency]

    # 生成未来时间序列
    future_times = []
    current_time = last_timestamp

    for i in range(pred_len):
        current_time += time_delta
        future_times.append(current_time)

    return pd.Series(future_times)


def predict_future_with_kronos(data_file, model_name="NeoQuasar/Kronos-small",
                               lookback=400, pred_len=120, frequency='H1'):
    """
    使用Kronos预测未来走势 - 修改自官方示例

    参数说明:
    data_file: str, 历史数据CSV文件路径
    model_name: str, 使用的Kronos模型
    lookback: int, 历史数据窗口长度（建议不超过512）
    pred_len: int, 预测的未来周期数
    frequency: str, 数据频率

    返回:
    dict, 包含预测结果的字典
    """

    print("🚀 基于官方示例的Kronos未来预测")
    print("=" * 50)
    print(f"模型: {model_name}")
    print(f"历史窗口: {lookback}")
    print(f"预测长度: {pred_len}")
    print(f"数据频率: {frequency}")
    print()

    # 步骤1: 加载模型和分词器（与官方示例相同）
    print("📥 加载预训练模型...")
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

        model = Kronos.from_pretrained(model_name)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 步骤2: 初始化预测器（与官方示例相同）
    print("🔧 初始化预测器...")
    # predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    # 步骤3: 加载数据（与官方示例相同）
    print("📊 加载历史数据...")
    try:
        df = pd.read_csv(data_file)
        df['timestamps'] = pd.to_datetime(df['timestamps'])

        # 确保数据按时间排序
        df = df.sort_values('timestamps').reset_index(drop=True)

        print(f"✅ 数据加载成功，共 {len(df)} 条记录")
        print(f"时间范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")

        # 检查必需列
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据缺少必需列: {required_cols}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

    # 步骤4: 准备输入数据（这里是关键修改）
    print("🎯 准备预测输入...")

    # 关键修改1: 使用最新的历史数据作为输入，而非中间段数据
    total_length = len(df)

    # 确保不超出数据范围
    start_idx = max(0, total_length - lookback)
    end_idx = total_length

    # 输入数据：最近的历史数据（而非官方示例中的前lookback条）
    if 'volume' in df.columns and 'amount' in df.columns:
        x_df = df.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume', 'amount']]
        print("✓ 使用完整的OHLCVA数据")
    else:
        x_df = df.iloc[start_idx:end_idx][['open', 'high', 'low', 'close']]
        print("✓ 使用OHLC基础数据")

    x_timestamp = df.iloc[start_idx:end_idx]['timestamps']

    print(f"输入数据范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")

    # 关键修改2: 生成未来时间戳，而非使用已知历史时间戳
    print("⏰ 生成未来时间戳...")
    last_known_time = df['timestamps'].iloc[-1]
    y_timestamp = generate_future_timestamps(last_known_time, pred_len, frequency)

    print(f"预测时间范围: {y_timestamp.iloc[0]} 到 {y_timestamp.iloc[-1]}")

    # 步骤5: 执行预测（与官方示例基本相同，但参数可调优）
    print("🔮 开始预测...")
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,  # 关键：这里是未来时间戳！
            pred_len=pred_len,
            T=1.0,  # 温度参数：可以调整预测的保守/激进程度 默认: 1.0
            top_p=0.6,  # 核采样参数：控制预测多样性 默认: 0.9
            sample_count=3  # 采样次数：多次采样提高稳定性 默认: 1,推荐 1-3
        )

        print("✅ 预测完成！")
        print(f"预测结果形状: {pred_df.shape}")
        print("\n预测结果前5行：")
        print(pred_df.head())

        return {
            'prediction': pred_df,
            'input_data': x_df,
            'input_timestamps': x_timestamp,
            'prediction_timestamps': y_timestamp,
            'last_known_price': df['close'].iloc[-1],
            'config': {
                'model_name': model_name,
                'lookback': lookback,
                'pred_len': pred_len,
                'frequency': frequency
            }
        }

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return None


def plot_prediction_results(result, save_path="kronos_future_prediction.png"):
    """
    绘制预测结果图表

    参数说明:
    result: dict, predict_future_with_kronos 返回的结果
    save_path: str, 图表保存路径
    """

    if result is None:
        print("❌ 无预测结果可绘制")
        return

    pred_df = result['prediction']
    input_df = result['input_data']
    input_timestamps = result['input_timestamps']
    pred_timestamps = result['prediction_timestamps']

    plt.figure(figsize=(16, 12))

    # 子图1: 价格走势对比
    plt.subplot(3, 1, 1)

    # 显示最近的历史数据（避免图表过于拥挤）
    recent_data = input_df.tail(200)
    recent_timestamps = input_timestamps.tail(200)

    # 历史价格线
    plt.plot(recent_timestamps, recent_data['close'],
             'b-', label='history CLOSE', linewidth=1.5, alpha=0.8)

    # 预测价格线
    plt.plot(pred_timestamps, pred_df['close'],
             'r-', label='prediction CLOSE', linewidth=2)

    # 添加预测起点分界线
    plt.axvline(x=input_timestamps.iloc[-1], color='gray',
                linestyle='--', alpha=0.7, label='Prediction Start')

    plt.title('Kronos prediction - CLOSE', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: OHLC完整信息
    plt.subplot(3, 1, 2)

    # 历史价格区间
    plt.fill_between(recent_timestamps,
                     recent_data['low'],
                     recent_data['high'],
                     alpha=0.3, color='blue', label='History Price Area')

    # 预测价格区间
    plt.fill_between(pred_timestamps,
                     pred_df['low'],
                     pred_df['high'],
                     alpha=0.4, color='red', label='Prediction Price Area')

    # 预测的开盘收盘价
    plt.plot(pred_timestamps, pred_df['open'],
             'g--', label='pre OPEN', alpha=0.8)
    plt.plot(pred_timestamps, pred_df['close'],
             'r-', label='pre CLOSE', linewidth=2)

    plt.axvline(x=input_timestamps.iloc[-1], color='gray',
                linestyle='--', alpha=0.7)

    plt.title('Kronos Prediction - OHLC', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 预测统计信息
    plt.subplot(3, 1, 3)

    # 计算价格变化百分比
    price_changes = (pred_df['close'].pct_change() * 100).dropna()

    # 绘制价格变化直方图
    plt.hist(price_changes, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=price_changes.mean(), color='red',
                linestyle='--', label=f'AVG: {price_changes.mean():.2f}%')

    plt.title('Prediction Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Price changes (%)')
    plt.ylabel('Times')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 图表已保存至: {save_path}")


def main():
    """
    主函数：执行完整的未来预测流程
    """

    print("🎯 Kronos未来走势预测")
    print("基于官方示例脚本修改版本")
    print("=" * 50)

    # 配置参数（请根据您的实际情况修改）
    config = {
        "data_file": "./data/XAUUSDH1_utf8.csv",  # 您的数据文件路径
        "model_name": "NeoQuasar/Kronos-small",  # 推荐从small开始
        "lookback": 400,  # 历史数据窗口
        "pred_len": 120,  # 预测未来120个周期
        "frequency": "H1"  # 数据频率，请匹配您的数据
    }

    print("📋 预测配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # 执行预测
    result = predict_future_with_kronos(**config)

    if result is not None:
        # 生成图表
        print("📊 生成预测图表...")
        plot_prediction_results(result)

        # 保存预测结果到CSV
        output_file = "kronos_future_prediction.csv"
        pred_df = result['prediction'].copy()
        pred_df['timestamps'] = result['prediction_timestamps']
        pred_df.to_csv(output_file, index=False)
        print(f"💾 预测结果已保存: {output_file}")

        # 显示预测摘要
        pred_df = result['prediction']
        pred_timestamps = result['prediction_timestamps']
        last_price = result['last_known_price']

        print("\n📈 预测摘要:")
        print(f"最后已知价格: {last_price:.4f}")
        print(f"预测时间跨度: {pred_timestamps.iloc[0]} 到 {pred_timestamps.iloc[-1]}")
        print(f"预测最终价格: {pred_df['close'].iloc[-1]:.4f}")
        print(f"预测总体变化: {((pred_df['close'].iloc[-1] / last_price) - 1) * 100:.2f}%")
        print(f"预测最高价: {pred_df['high'].max():.4f}")
        print(f"预测最低价: {pred_df['low'].min():.4f}")

    else:
        print("❌ 预测失败，请检查配置和数据")


if __name__ == "__main__":
    main()
