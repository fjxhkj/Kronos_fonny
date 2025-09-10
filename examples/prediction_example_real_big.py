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


def plot_prediction_results(result, save_path="kronos_future_prediction.png",
                            history_display_ratio=0.3, y_axis_expand_ratio=0.1):
    """
    绘制预测结果图表 - 仅保留 OHLC 图

    参数说明:
    result: dict, predict_future_with_kronos 返回的结果
    save_path: str, 图表保存路径
    history_display_ratio: float, 历史数据显示比例 (0.0-1.0)
    y_axis_expand_ratio: float, Y轴扩展比例
    """

    if result is None:
        print("❌ 无预测结果可绘制")
        return

    pred_df = result['prediction']
    input_df = result['input_data']
    input_timestamps = result['input_timestamps']
    pred_timestamps = result['prediction_timestamps']

    # 创建图表，使用更大的尺寸以便更好地显示OHLC数据
    plt.figure(figsize=(16, 10))

    # 计算要显示的历史数据量（减少历史数据显示比例）
    history_display_count = max(1, int(len(input_df) * history_display_ratio))
    recent_data = input_df.tail(history_display_count)
    recent_timestamps = input_timestamps.tail(history_display_count)

    # 绘制历史数据的OHLC K线图
    for i in range(len(recent_data)):
        timestamp = recent_timestamps.iloc[i]
        open_price = recent_data['open'].iloc[i]
        high_price = recent_data['high'].iloc[i]
        low_price = recent_data['low'].iloc[i]
        close_price = recent_data['close'].iloc[i]

        # K线颜色：涨为绿色，跌为红色
        color = 'green' if close_price >= open_price else 'red'

        # 绘制高低价线
        plt.plot([timestamp, timestamp], [low_price, high_price],
                 color=color, linewidth=1, alpha=0.8)

        # 绘制开盘收盘价矩形
        body_height = abs(close_price - open_price)
        if body_height > 0:
            bottom = min(open_price, close_price)
            plt.bar(timestamp, body_height, bottom=bottom,
                    color=color, alpha=0.7, width=pd.Timedelta(hours=0.8))

    # 绘制预测数据的OHLC K线图
    for i in range(len(pred_df)):
        timestamp = pred_timestamps.iloc[i]
        open_price = pred_df['open'].iloc[i]
        high_price = pred_df['high'].iloc[i]
        low_price = pred_df['low'].iloc[i]
        close_price = pred_df['close'].iloc[i]

        # 预测K线使用不同的颜色和透明度
        color = 'lightgreen' if close_price >= open_price else 'lightcoral'

        # 绘制高低价线
        plt.plot([timestamp, timestamp], [low_price, high_price],
                 color=color, linewidth=1.5, alpha=0.9)

        # 绘制开盘收盘价矩形
        body_height = abs(close_price - open_price)
        if body_height > 0:
            bottom = min(open_price, close_price)
            plt.bar(timestamp, body_height, bottom=bottom,
                    color=color, alpha=0.8, width=pd.Timedelta(hours=0.8))

    # 添加预测起点分界线
    plt.axvline(x=input_timestamps.iloc[-1], color='gray',
                linestyle='--', linewidth=2, alpha=0.7, label='pre start')

    # 计算Y轴范围并增加显示比例
    all_prices = []
    all_prices.extend(recent_data['high'].tolist())
    all_prices.extend(recent_data['low'].tolist())
    all_prices.extend(pred_df['high'].tolist())
    all_prices.extend(pred_df['low'].tolist())

    price_min = min(all_prices)
    price_max = max(all_prices)
    price_range = price_max - price_min

    # 增加Y轴显示范围
    y_margin = price_range * y_axis_expand_ratio
    plt.ylim(price_min - y_margin, price_max + y_margin)

    # 设置标题和标签
    plt.title('Kronos Future Trend Prediction - OHLC Chart',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='his up'),
        Patch(facecolor='red', alpha=0.7, label='his dw'),
        Patch(facecolor='lightgreen', alpha=0.8, label='pre up'),
        Patch(facecolor='lightcoral', alpha=0.8, label='pre dw'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='pre start')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 自动格式化x轴日期显示
    plt.xticks(rotation=45)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 OHLC图表已保存至: {save_path}")

    # 输出图表配置信息
    print(f"📈 图表配置:")
    print(f"  历史数据显示比例: {history_display_ratio * 100:.1f}%")
    print(f"  Y轴扩展比例: {y_axis_expand_ratio * 100:.1f}%")
    print(f"  显示的历史数据点数: {history_display_count}")
    print(f"  预测数据点数: {len(pred_df)}")


def main():
    """
    主函数：执行完整的未来预测流程
    """

    print("🎯 Kronos未来走势预测")
    print("基于官方示例脚本修改版本 - 仅OHLC图表")
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
        # 生成OHLC图表（减少历史数据显示，增加Y轴比例）
        print("📊 生成OHLC预测图表...")
        plot_prediction_results(
            result,
            save_path="kronos_future_ohlc_prediction.png",
            history_display_ratio=0.25,  # 只显示25%的历史数据
            y_axis_expand_ratio=0.15  # Y轴扩展15%
        )

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
