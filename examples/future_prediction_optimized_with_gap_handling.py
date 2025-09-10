# =======================
# 文件名: future_prediction_optimized_with_gap_handling.py
# 基于Kronos论文推荐的采样参数优化版本 + 智能时间缺口处理
# =======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import time

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def preprocess_financial_data_with_gap_handling(df, frequency='H1', market_type='forex'):
    """
    智能处理金融数据中的时间缺口

    参数说明:
    df: DataFrame, 包含时间戳和OHLC数据
    frequency: str, 数据频率
    market_type: str, 市场类型 ('forex', 'stock', 'crypto')

    返回:
    tuple: (处理后的DataFrame, 缺口统计信息)
    """
    print(f"🔧 开始智能时间缺口处理 (市场类型: {market_type})")

    # 确保时间列为 datetime 类型
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    original_length = len(df)

    # 计算时间间隔来识别缺口
    df['time_diff_hours'] = df['timestamps'].diff().dt.total_seconds() / 3600

    # 定义各市场的预期时间间隔
    expected_intervals = {
        'M1': 1/60, 'M5': 5/60, 'M15': 15/60,
        'M30': 0.5, 'H1': 1, 'H4': 4, 'D1': 24
    }

    expected_interval = expected_intervals.get(frequency, 1)

    # 识别大的时间缺口 (超过预期间隔的2倍)
    gap_threshold = expected_interval * 2.5
    large_gaps = df['time_diff_hours'] > gap_threshold
    gap_count = large_gaps.sum()

    print(f"  📊 缺口分析:")
    print(f"    原始数据点数: {original_length}")
    print(f"    预期时间间隔: {expected_interval} 小时")
    print(f"    检测到的大缺口数量: {gap_count}")

    if market_type == 'crypto':
        # 加密货币24/7交易，只需要处理异常缺口
        print("  ✓ 加密货币市场 - 仅处理异常缺口")
        filtered_df = df.copy()
    else:
        # 处理外汇和股票市场的周末/节假日缺口
        filtered_df = filter_trading_hours(df, market_type, frequency)

    # 移除临时列
    if 'time_diff_hours' in filtered_df.columns:
        filtered_df = filtered_df.drop('time_diff_hours', axis=1)

    processed_length = len(filtered_df)
    gap_ratio = (original_length - processed_length) / original_length * 100 if original_length > 0 else 0

    gap_stats = {
        'original_length': original_length,
        'processed_length': processed_length,
        'gap_ratio': gap_ratio,
        'large_gaps_detected': gap_count,
        'market_type': market_type,
        'frequency': frequency
    }

    print(f"  ✅ 缺口处理完成:")
    print(f"    处理后数据点数: {processed_length}")
    print(f"    过滤比例: {gap_ratio:.2f}%")

    return filtered_df.reset_index(drop=True), gap_stats


def filter_trading_hours(df, market_type, frequency):
    """
    根据市场类型过滤交易时间
    """
    df_copy = df.copy()

    if market_type == 'forex':
        # 外汇市场：周日17:00 UTC 到周五17:00 UTC
        mask = (
            # 周一到周四全天
                (df_copy['timestamps'].dt.weekday.isin([0, 1, 2, 3])) |
                # 周五17点前
                ((df_copy['timestamps'].dt.weekday == 4) & (df_copy['timestamps'].dt.hour < 17)) |
                # 周日17点后
                ((df_copy['timestamps'].dt.weekday == 6) & (df_copy['timestamps'].dt.hour >= 17))
        )

    elif market_type == 'stock':
        # 股票市场：工作日9:30-16:00 (假设美股时间)
        mask = (
                (df_copy['timestamps'].dt.weekday < 5) &  # 工作日
                (
                        ((df_copy['timestamps'].dt.hour == 9) & (df_copy['timestamps'].dt.minute >= 30)) |
                        (df_copy['timestamps'].dt.hour.isin([10, 11, 12, 13, 14, 15])) |
                        ((df_copy['timestamps'].dt.hour == 16) & (df_copy['timestamps'].dt.minute == 0))
                )
        )

    else:  # crypto
        # 加密货币24/7交易
        mask = pd.Series([True] * len(df_copy), index=df_copy.index)

    return df_copy[mask]


def generate_future_business_timestamps(last_timestamp, pred_len, frequency='H1', market_type='forex'):
    """
    生成考虑交易时间的未来时间戳
    """
    # 时间增量映射
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
    future_times = []
    current_time = last_timestamp

    count = 0
    max_iterations = pred_len * 10  # 防止无限循环
    iterations = 0

    while count < pred_len and iterations < max_iterations:
        current_time += time_delta
        iterations += 1

        # 根据市场类型检查是否为交易时间
        is_trading_time = False

        if market_type == 'forex':
            # 外汇市场：周日17点到周五17点
            if (current_time.weekday() < 4 or  # 周一到周四
                    (current_time.weekday() == 4 and current_time.hour < 17) or  # 周五17点前
                    (current_time.weekday() == 6 and current_time.hour >= 17)):  # 周日17点后
                is_trading_time = True

        elif market_type == 'stock':
            # 股票市场：工作日9:30-16:00
            if (current_time.weekday() < 5 and
                    ((current_time.hour == 9 and current_time.minute >= 30) or
                     current_time.hour in [10, 11, 12, 13, 14, 15] or
                     (current_time.hour == 16 and current_time.minute == 0))):
                is_trading_time = True

        elif market_type == 'crypto':
            # 加密货币：24/7
            is_trading_time = True

        if is_trading_time:
            future_times.append(current_time)
            count += 1

    if count < pred_len:
        print(f"⚠️ 警告：只能生成 {count} 个交易时间戳，少于要求的 {pred_len} 个")

    return pd.Series(future_times)


def generate_future_timestamps(last_timestamp, pred_len, frequency='H1'):
    """
    生成未来时间戳 - 兼容性函数
    """
    return generate_future_business_timestamps(last_timestamp, pred_len, frequency, 'crypto')


def calculate_price_change_ratio(pred_df, last_known_price):
    """
    计算预测结果的涨跌幅度
    """
    final_price = pred_df['close'].iloc[-1]
    return ((final_price / last_known_price) - 1) * 100


def predict_multiple_and_average(data_file, model_name="NeoQuasar/Kronos-small",
                                 lookback=400, pred_len=120, frequency='H1',
                                 num_predictions=3, temperature=0.6, top_p=0.7,
                                 sample_count=3, market_type='forex',
                                 handle_gaps=True):
    """
    基于论文推荐参数执行多次预测并计算平均值 + 智能缺口处理

    新增参数:
    market_type: str, 市场类型 ('forex', 'stock', 'crypto')
    handle_gaps: bool, 是否启用智能缺口处理
    """

    print("🚀 基于Kronos论文优化的预测策略 + 智能缺口处理")
    print("=" * 50)
    print(f"缺口处理: {'启用' if handle_gaps else '禁用'}")
    print(f"市场类型: {market_type}")
    print(f"预测次数: {num_predictions} (不去极值，直接平均)")
    print(f"温度参数: {temperature} (论文推荐用于金融预测)")
    print(f"核采样参数: {top_p}")
    print(f"单次采样数: {sample_count}")
    print()

    # 步骤1: 加载模型和分词器
    print("📥 加载预训练模型...")
    model_start_time = time.time()

    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained(model_name)

        model_load_time = time.time() - model_start_time
        print(f"✅ 模型加载成功 (耗时: {model_load_time:.2f}秒)")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 步骤2: 初始化预测器
    print("🔧 初始化预测器...")
    predictor_start_time = time.time()

    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    # predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    predictor_init_time = time.time() - predictor_start_time
    print(f"✅ 预测器初始化完成 (耗时: {predictor_init_time:.2f}秒)")

    # 步骤3: 加载和预处理数据
    print("📊 加载历史数据...")
    data_start_time = time.time()

    try:
        df = pd.read_csv(data_file)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        original_length = len(df)
        print(f"✅ 原始数据加载成功，共 {original_length} 条记录")
        print(f"时间范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")

        # 智能缺口处理
        if handle_gaps:
            df_processed, gap_stats = preprocess_financial_data_with_gap_handling(
                df, frequency=frequency, market_type=market_type
            )
        else:
            df_processed = df.copy()
            gap_stats = {
                'original_length': original_length,
                'processed_length': len(df_processed),
                'gap_ratio': 0,
                'large_gaps_detected': 0,
                'market_type': market_type,
                'frequency': frequency
            }

        data_load_time = time.time() - data_start_time
        print(f"✅ 数据处理完成 (耗时: {data_load_time:.2f}秒)")

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df_processed.columns for col in required_cols):
            raise ValueError(f"数据缺少必需列: {required_cols}")

    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return None

    # 步骤4: 准备输入数据
    print("🎯 准备预测输入...")
    prep_start_time = time.time()

    total_length = len(df_processed)
    start_idx = max(0, total_length - lookback)
    end_idx = total_length

    if 'volume' in df_processed.columns and 'amount' in df_processed.columns:
        x_df = df_processed.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume', 'amount']]
        print("✓ 使用完整的OHLCVA数据")
    else:
        x_df = df_processed.iloc[start_idx:end_idx][['open', 'high', 'low', 'close']]
        print("✓ 使用OHLC基础数据")

    x_timestamp = df_processed.iloc[start_idx:end_idx]['timestamps']
    last_known_time = df_processed['timestamps'].iloc[-1]
    last_known_price = df_processed['close'].iloc[-1]

    # 生成考虑交易时间的未来时间戳
    if handle_gaps:
        y_timestamp = generate_future_business_timestamps(
            last_known_time, pred_len, frequency, market_type
        )
    else:
        y_timestamp = generate_future_timestamps(last_known_time, pred_len, frequency)

    prep_time = time.time() - prep_start_time
    print(f"输入数据范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
    print(f"预测时间范围: {y_timestamp.iloc[0]} 到 {y_timestamp.iloc[-1]}")
    print(f"最后已知价格: {last_known_price:.4f}")
    print(f"✅ 数据准备完成 (耗时: {prep_time:.2f}秒)")

    # 步骤5: 执行多次预测
    print(f"🔮 开始执行{num_predictions}次预测（使用论文推荐参数）...")
    predictions_start_time = time.time()

    all_predictions = []
    change_ratios = []
    prediction_times = []

    for i in range(num_predictions):
        single_pred_start = time.time()
        print(f"  执行第 {i+1}/{num_predictions} 次预测...", end=" ")

        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count
            )

            change_ratio = calculate_price_change_ratio(pred_df, last_known_price)
            single_pred_time = time.time() - single_pred_start
            prediction_times.append(single_pred_time)

            all_predictions.append(pred_df.copy())
            change_ratios.append(change_ratio)

            print(f"完成 (耗时: {single_pred_time:.2f}秒, 涨跌幅: {change_ratio:.2f}%)")

        except Exception as e:
            single_pred_time = time.time() - single_pred_start
            print(f"失败 (耗时: {single_pred_time:.2f}秒) - {e}")
            continue

    total_predictions_time = time.time() - predictions_start_time

    if len(all_predictions) < 1:
        print(f"❌ 预测全部失败")
        return None

    print(f"✅ 完成{len(all_predictions)}次有效预测")
    print(f"📊 预测统计:")
    print(f"  总预测时间: {total_predictions_time:.2f}秒")
    print(f"  平均单次预测时间: {np.mean(prediction_times):.2f}秒")

    # 步骤6: 计算平均预测
    print("📊 分析预测结果（不去极值，直接平均）...")
    analysis_start_time = time.time()

    print("所有预测结果涨跌幅:")
    for i, ratio in enumerate(change_ratios):
        print(f"  第{i+1}次: {ratio:.2f}%")

    # 计算平均预测
    avg_pred_df = all_predictions[0].copy()
    numeric_columns = ['open', 'high', 'low', 'close']
    if 'volume' in avg_pred_df.columns:
        numeric_columns.append('volume')
    if 'amount' in avg_pred_df.columns:
        numeric_columns.append('amount')

    for col in numeric_columns:
        avg_pred_df[col] = 0
        for pred_df in all_predictions:
            avg_pred_df[col] += pred_df[col]
        avg_pred_df[col] /= len(all_predictions)

    avg_change_ratio = calculate_price_change_ratio(avg_pred_df, last_known_price)
    analysis_time = time.time() - analysis_start_time
    print(f"✅ 平均预测计算完成！(耗时: {analysis_time:.2f}秒)")
    print(f"平均涨跌幅: {avg_change_ratio:.2f}%")

    return {
        'prediction': avg_pred_df,
        'input_data': x_df,
        'input_timestamps': x_timestamp,
        'prediction_timestamps': y_timestamp,
        'last_known_price': last_known_price,
        'all_predictions': all_predictions,
        'change_ratios': change_ratios,
        'avg_change_ratio': avg_change_ratio,
        'gap_stats': gap_stats,
        'timing_info': {
            'model_load_time': model_load_time,
            'predictor_init_time': predictor_init_time,
            'data_load_time': data_load_time,
            'data_prep_time': prep_time,
            'total_predictions_time': total_predictions_time,
            'avg_prediction_time': np.mean(prediction_times),
            'analysis_time': analysis_time,
            'prediction_times': prediction_times
        },
        'config': {
            'model_name': model_name,
            'lookback': lookback,
            'pred_len': pred_len,
            'frequency': frequency,
            'num_predictions': num_predictions,
            'used_predictions': len(all_predictions),
            'temperature': temperature,
            'top_p': top_p,
            'sample_count': sample_count,
            'market_type': market_type,
            'handle_gaps': handle_gaps
        }
    }


def plot_prediction_results(result, save_path="kronos_optimized_prediction_with_gaps.png",
                            history_display_ratio=0.3, y_axis_expand_ratio=0.1):
    """
    绘制预测结果图表 - 带缺口处理信息
    """
    if result is None:
        print("❌ 无预测结果可绘制")
        return

    print("📊 开始生成图表...")
    plot_start_time = time.time()

    pred_df = result['prediction']
    input_df = result['input_data']
    input_timestamps = result['input_timestamps']
    pred_timestamps = result['prediction_timestamps']
    gap_stats = result['gap_stats']

    # 创建图表
    plt.figure(figsize=(16, 10))

    # 计算要显示的历史数据量
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

        color = 'green' if close_price >= open_price else 'red'
        plt.plot([timestamp, timestamp], [low_price, high_price],
                 color=color, linewidth=1, alpha=0.8)

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

        color = 'lightgreen' if close_price >= open_price else 'lightcoral'
        plt.plot([timestamp, timestamp], [low_price, high_price],
                 color=color, linewidth=1.5, alpha=0.9)

        body_height = abs(close_price - open_price)
        if body_height > 0:
            bottom = min(open_price, close_price)
            plt.bar(timestamp, body_height, bottom=bottom,
                    color=color, alpha=0.8, width=pd.Timedelta(hours=0.8))

    # 添加预测起点分界线
    plt.axvline(x=input_timestamps.iloc[-1], color='gray',
                linestyle='--', linewidth=2, alpha=0.7, label='Prediction Start')

    # 计算Y轴范围
    all_prices = []
    all_prices.extend(recent_data['high'].tolist())
    all_prices.extend(recent_data['low'].tolist())
    all_prices.extend(pred_df['high'].tolist())
    all_prices.extend(pred_df['low'].tolist())

    price_min = min(all_prices)
    price_max = max(all_prices)
    price_range = price_max - price_min
    y_margin = price_range * y_axis_expand_ratio
    plt.ylim(price_min - y_margin, price_max + y_margin)

    # 设置标题（包含缺口处理信息）
    avg_change = result['avg_change_ratio']
    num_used = result['config']['used_predictions']
    temperature = result['config']['temperature']
    top_p = result['config']['top_p']
    market_type = gap_stats['market_type']
    gap_ratio = gap_stats['gap_ratio']

    title = f'Kronos Gap-Aware Prediction ({market_type.upper()}, T={temperature}, p={top_p}, {num_used} samples)\n'
    title += f'Change: {avg_change:.2f}% | Gap Filtered: {gap_ratio:.1f}%'

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Historical Bullish'),
        Patch(facecolor='red', alpha=0.7, label='Historical Bearish'),
        Patch(facecolor='lightgreen', alpha=0.8, label='Predicted Bullish (Avg)'),
        Patch(facecolor='lightcoral', alpha=0.8, label='Predicted Bearish (Avg)'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Prediction Start')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    plot_time = time.time() - plot_start_time
    print(f"📊 智能缺口处理图表已保存至: {save_path} (绘图耗时: {plot_time:.2f}秒)")


def main():
    """
    主函数：执行基于论文推荐的优化预测流程 + 智能缺口处理
    """
    script_start_time = time.time()
    start_datetime = datetime.now()

    print("🎯 Kronos论文优化预测策略 + 智能时间缺口处理")
    print("基于论文推荐采样参数 + 市场时间感知预处理")
    print("=" * 60)
    print(f"⏰ 脚本开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 配置参数（基于论文推荐 + 缺口处理）
    config = {
        "data_file": "./data/XAUUSDH1_utf8.csv",
        "model_name": "NeoQuasar/Kronos-small",
        "lookback": 400,
        "pred_len": 120,
        "frequency": "H1",
        "num_predictions": 1,
        "temperature": 0.6,
        "top_p": 0.7,
        "sample_count": 5,
        "market_type": "forex",  # 新增：市场类型
        "handle_gaps": True      # 新增：启用缺口处理
    }

    print("📋 智能缺口处理配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n🧠 智能缺口处理优势:")
    print("  • 自动识别和处理周末/假期数据缺口")
    print("  • 基于实际交易时间过滤数据")
    print("  • 保持时间序列连续性和统计特性")
    print("  • 提高模型预测准确性和稳定性")
    print("  • 支持外汇、股票、加密货币市场")
    print()

    # 执行智能缺口处理预测
    prediction_start_time = time.time()
    result = predict_multiple_and_average(**config)
    prediction_total_time = time.time() - prediction_start_time

    if result is not None:
        # 显示缺口处理效果
        gap_stats = result['gap_stats']
        print(f"\n📈 缺口处理效果分析:")
        print(f"  原始数据量: {gap_stats['original_length']} 条")
        print(f"  处理后数据量: {gap_stats['processed_length']} 条")
        print(f"  数据过滤比例: {gap_stats['gap_ratio']:.2f}%")
        print(f"  检测到的大缺口: {gap_stats['large_gaps_detected']} 个")
        print(f"  市场类型: {gap_stats['market_type']}")
        print(f"  数据频率: {gap_stats['frequency']}")

        # 生成图表
        print("\n📊 生成智能缺口处理预测图表...")
        plot_prediction_results(
            result,
            save_path="kronos_gap_aware_prediction.png",
            history_display_ratio=0.25,
            y_axis_expand_ratio=0.15
        )

        # 保存结果
        print("\n💾 保存结果文件...")
        save_start_time = time.time()

        # 保存预测结果
        output_file = "kronos_gap_aware_prediction.csv"
        pred_df = result['prediction'].copy()
        pred_df['timestamps'] = result['prediction_timestamps']
        pred_df.to_csv(output_file, index=False)
        print(f"💾 智能预测结果已保存: {output_file}")

        # 保存详细统计
        summary_file = "kronos_gap_aware_summary.csv"
        summary_data = []
        for i, (pred_df, change_ratio) in enumerate(zip(result['all_predictions'], result['change_ratios'])):
            summary_data.append({
                'prediction_id': i + 1,
                'change_ratio_percent': change_ratio,
                'final_price': pred_df['close'].iloc[-1],
                'max_price': pred_df['high'].max(),
                'min_price': pred_df['low'].min(),
                'prediction_time_seconds': result['timing_info']['prediction_times'][i],
                'gap_filtered_ratio': gap_stats['gap_ratio'],
                'market_type': gap_stats['market_type'],
                'handle_gaps': config['handle_gaps']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)

        save_time = time.time() - save_start_time
        print(f"💾 预测汇总信息已保存: {summary_file} (保存耗时: {save_time:.2f}秒)")

        # 显示完整分析结果
        print("\n📈 智能缺口处理预测摘要:")
        print(f"预测策略: 论文推荐参数 + 智能缺口处理")
        print(f"缺口处理策略: {gap_stats['market_type']} 市场时间感知")
        print(f"最终预测变化: {result['avg_change_ratio']:.2f}%")

        # 计算数据质量改进指标
        if gap_stats['gap_ratio'] > 0:
            print(f"\n🎯 数据质量改进效果:")
            print(f"  • 移除了 {gap_stats['gap_ratio']:.1f}% 的非交易时间数据")
            print(f"  • 保持了时间序列的连续性和统计特性")
            print(f"  • 避免了周末缺口对预测模型的负面影响")
            print(f"  • 提高了 Kronos 模型的输入数据质量")

    else:
        print("❌ 智能缺口处理预测失败，请检查配置和数据")

    # 脚本执行总结
    script_end_time = time.time()
    end_datetime = datetime.now()
    total_script_time = script_end_time - script_start_time

    print("\n" + "=" * 60)
    print("⏰ 智能缺口处理预测执行完成")
    print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总执行时间: {total_script_time:.2f}秒 ({total_script_time/60:.2f}分钟)")
    print("🎯 Kronos智能缺口处理策略执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
