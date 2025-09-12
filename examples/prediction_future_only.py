# =======================
# 文件名: future_prediction_optimized.py
# 基于推荐的采样参数优化版本
# =======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import time
import os
import shutil
from huggingface_hub import snapshot_download

# 尝试导入 torch 并检测 CUDA 可用性
try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

# 确保将项目根目录（脚本父目录的父目录）加入 sys.path，
# 无论从哪里运行脚本都能正确导入项目内模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model import Kronos, KronosTokenizer, KronosPredictor


def generate_future_timestamps(last_timestamp, pred_len, frequency="H1"):
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
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
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


def calculate_price_change_ratio(pred_df, last_known_price):
    """
    计算预测结果的涨跌幅度

    参数说明:
    pred_df: DataFrame, 预测结果
    last_known_price: float, 最后已知价格

    返回:
    float, 总体涨跌幅度
    """
    final_price = pred_df["close"].iloc[-1]
    return ((final_price / last_known_price) - 1) * 100


def predict_multiple_and_average(
    data_file,
    model_name,
    lookback,
    pred_len,
    frequency,
    num_predictions,
    temperature,
    top_p,
    sample_count,
):
    """
    基于推荐参数执行多次预测并计算平均值

    参数说明:
    data_file: str, 历史数据CSV文件路径
    model_name: str, 使用的Kronos模型
    lookback: int, 历史数据窗口长度
    pred_len: int, 预测的未来周期数
    frequency: str, 数据频率
    num_predictions: int, 预测次数（推荐3次）
    temperature: float, 温度参数（推荐0.6用于金融预测）
    top_p: float, 核采样参数（推荐0.7）
    sample_count: int, 每次预测的采样次数（推荐3）

    返回:
    dict, 包含平均预测结果的字典
    """

    print("🚀 优化的预测策略")
    print("=" * 50)
    print(f"预测次数: {num_predictions} (不去极值，直接平均)")
    print(f"温度参数: {temperature} (推荐用于金融预测)")
    print(f"核采样参数: {top_p}")
    print(f"单次采样数: {sample_count}")
    print()

    # 步骤1: 加载模型和分词器（只需加载一次）
    print("📥 加载预训练模型...")
    model_start_time = time.time()

    try:
        # 优化缓存策略：
        # 1) 首先检查系统是否存在 Z: 盘（内存盘），若存在且 Z:\model_cache 有内容则直接使用（无需覆盖或复制）
        # 2) 否则检查项目目录下的持久备份（model_cache_backup），若存在且有内容则直接使用（无需重新下载）
        # 3) 若两者都不存在，则根据 Z: 是否可用决定下载目标（优先 Z:），下载完成后在项目目录保留备份以防重启丢失
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        backup_cache = os.path.join(project_root, "model_cache_backup")
        z_root_exists = os.path.exists("Z:\\") and os.path.isdir("Z:\\")
        z_cache = r"Z:\model_cache"

        # 设置 HTTP 代理（如有需要）
        proxy = "http://127.0.0.1:11082"
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("http_proxy", proxy)
        os.environ.setdefault("https_proxy", proxy)

        cache_dir_to_use = None

        # 优先：如果 Z:\model_cache 存在且非空，直接使用
        if z_root_exists and os.path.exists(z_cache) and len(os.listdir(z_cache)) > 0:
            cache_dir_to_use = z_cache
            print(f"⚡ 使用 Z: 盘缓存 -> {z_cache} (无需覆盖)")
        # 次选：项目内持久备份存在且非空，使用之
        elif os.path.exists(backup_cache) and len(os.listdir(backup_cache)) > 0:
            cache_dir_to_use = backup_cache
            print(f"📁 使用项目内持久缓存 -> {backup_cache} (无需下载)")
        else:
            # 需要下载，优先下载到 Z:（如果可用），否则下载到项目备份目录
            download_target = z_cache if z_root_exists else backup_cache
            os.makedirs(download_target, exist_ok=True)
            print(f"📥 开始下载至 -> {download_target} ...")
            # 下载 tokenizer 与 model 到目标缓存
            local_tokenizer_dir = snapshot_download(
                repo_id="NeoQuasar/Kronos-Tokenizer-base",
                cache_dir=download_target,
                resume_download=True,
            )
            local_model_dir = snapshot_download(
                repo_id=model_name, cache_dir=download_target, resume_download=True
            )

            # 若下载到了 Z:，并且项目备份目录不存在或为空，则把下载结果备份回项目目录以持久化
            try:
                if download_target == z_cache:
                    if (
                        not os.path.exists(backup_cache)
                        or len(os.listdir(backup_cache)) == 0
                    ):
                        # 先清除原有备份（若存在），再复制
                        if os.path.exists(backup_cache):
                            try:
                                shutil.rmtree(backup_cache)
                            except Exception:
                                pass
                        shutil.copytree(z_cache, backup_cache)
                        print(f"💾 已将 Z: 缓存备份到项目目录 -> {backup_cache}")
                # 指定使用下载目标作为缓存目录
                cache_dir_to_use = download_target
            except Exception:
                # 备份失败不阻止继续运行
                cache_dir_to_use = download_target

        # 使用 snapshot_download 指定 cache_dir（若之前已存在缓存，这里通常很快返回本地路径）
        print(f"📥 确认并定位 tokenizer 与 model（cache_dir={cache_dir_to_use}）...")
        local_tokenizer_dir = snapshot_download(
            repo_id="NeoQuasar/Kronos-Tokenizer-base",
            cache_dir=cache_dir_to_use,
            resume_download=True,
        )
        local_model_dir = snapshot_download(
            repo_id=model_name, cache_dir=cache_dir_to_use, resume_download=True
        )

        tokenizer = KronosTokenizer.from_pretrained(
            local_tokenizer_dir, local_files_only=True
        )
        model = Kronos.from_pretrained(local_model_dir, local_files_only=True)

        model_load_time = time.time() - model_start_time
        print(f"✅ 模型加载成功 (耗时: {model_load_time:.2f}秒)")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 步骤2: 初始化预测器
    print("🔧 初始化预测器...")
    predictor_start_time = time.time()

    # 自动检测系统是否支持 GPU（优先使用第一个可用 GPU）
    try:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"

    print(f"⚙️ 使用设备: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    predictor_init_time = time.time() - predictor_start_time
    print(f"✅ 预测器初始化完成 (耗时: {predictor_init_time:.2f}秒)")

    # 步骤3: 加载数据
    print("📊 加载历史数据...")
    data_start_time = time.time()

    try:
        df = pd.read_csv(data_file)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)

        data_load_time = time.time() - data_start_time
        print(f"✅ 数据加载成功，共 {len(df)} 条记录 (耗时: {data_load_time:.2f}秒)")
        print(f"时间范围: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")

        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据缺少必需列: {required_cols}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

    # 步骤4: 准备输入数据
    print("🎯 准备预测输入...")
    prep_start_time = time.time()

    total_length = len(df)
    start_idx = max(0, total_length - lookback)
    end_idx = total_length

    if "volume" in df.columns and "amount" in df.columns:
        x_df = df.iloc[start_idx:end_idx][
            ["open", "high", "low", "close", "volume", "amount"]
        ]
        print("✓ 使用完整的OHLCVA数据")
    else:
        x_df = df.iloc[start_idx:end_idx][["open", "high", "low", "close"]]
        print("✓ 使用OHLC基础数据")

    x_timestamp = df.iloc[start_idx:end_idx]["timestamps"]
    last_known_time = df["timestamps"].iloc[-1]
    last_known_price = df["close"].iloc[-1]
    y_timestamp = generate_future_timestamps(last_known_time, pred_len, frequency)

    prep_time = time.time() - prep_start_time
    print(f"输入数据范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
    print(f"预测时间范围: {y_timestamp.iloc[0]} 到 {y_timestamp.iloc[-1]}")
    print(f"✅ 数据准备完成 (耗时: {prep_time:.2f}秒)")

    # 步骤5: 执行多次预测（使用优化参数）
    print(f"🔮 开始执行{num_predictions}次预测 ...")
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
                sample_count=sample_count,
            )

            # 计算这次预测的涨跌幅度
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
    if len(prediction_times) > 1:
        print(f"  最快预测时间: {min(prediction_times):.2f}秒")
        print(f"  最慢预测时间: {max(prediction_times):.2f}秒")

    # 步骤6: 直接计算平均值（不去极值）
    print("📊 分析预测结果（不去极值，直接平均）...")
    analysis_start_time = time.time()

    print("所有预测结果涨跌幅:")
    for i, ratio in enumerate(change_ratios):
        print(f"  第{i+1}次: {ratio:.2f}%")

    # 步骤7: 计算平均预测
    print("🧮 计算平均预测结果...")

    # 初始化平均值DataFrame
    avg_pred_df = all_predictions[0].copy()

    # 对所有数值列计算平均值
    numeric_columns = ["open", "high", "low", "close"]
    if "volume" in avg_pred_df.columns:
        numeric_columns.append("volume")
    if "amount" in avg_pred_df.columns:
        numeric_columns.append("amount")

    for col in numeric_columns:
        avg_pred_df[col] = 0

        # 累加所有预测结果
        for pred_df in all_predictions:
            avg_pred_df[col] += pred_df[col]

        # 计算平均值
        avg_pred_df[col] /= len(all_predictions)

    # 计算平均涨跌幅
    avg_change_ratio = calculate_price_change_ratio(avg_pred_df, last_known_price)

    analysis_time = time.time() - analysis_start_time
    print(f"✅ 平均预测计算完成！(耗时: {analysis_time:.2f}秒)")
    print(f"平均涨跌幅: {avg_change_ratio:.2f}%")

    return {
        "prediction": avg_pred_df,
        "input_data": x_df,
        "input_timestamps": x_timestamp,
        "prediction_timestamps": y_timestamp,
        "last_known_price": last_known_price,
        "all_predictions": all_predictions,
        "change_ratios": change_ratios,
        "avg_change_ratio": avg_change_ratio,
        "timing_info": {
            "model_load_time": model_load_time,
            "predictor_init_time": predictor_init_time,
            "data_load_time": data_load_time,
            "data_prep_time": prep_time,
            "total_predictions_time": total_predictions_time,
            "avg_prediction_time": np.mean(prediction_times),
            "analysis_time": analysis_time,
            "prediction_times": prediction_times,
        },
        "config": {
            "model_name": model_name,
            "lookback": lookback,
            "pred_len": pred_len,
            "frequency": frequency,
            "num_predictions": num_predictions,
            "used_predictions": len(all_predictions),
            "temperature": temperature,
            "top_p": top_p,
            "sample_count": sample_count,
        },
    }


def get_bar_width_for_frequency(frequency):
    """根据数据频率获取合适的柱状图宽度"""
    width_mapping = {
        "M1": pd.Timedelta(minutes=0.8),
        "M5": pd.Timedelta(minutes=4),
        "M15": pd.Timedelta(minutes=12),  # 15分钟图：12分钟宽度
        "M30": pd.Timedelta(minutes=24),
        "H1": pd.Timedelta(minutes=48),
        "H4": pd.Timedelta(hours=3.2),
        "D1": pd.Timedelta(hours=19.2),
    }
    return width_mapping.get(frequency, pd.Timedelta(hours=0.8))


def plot_prediction_results_adaptive(
    result,
    save_path,
    history_display_ratio,
    y_axis_expand_ratio,
):
    """自适应绘制预测结果 - 解决15分钟图蜡烛太宽的问题"""

    if result is None:
        print("❌ 无预测结果可绘制")
        return

    print("📊 开始生成自适应图表...")
    plot_start_time = time.time()

    pred_df = result["prediction"]
    input_df = result["input_data"]
    input_timestamps = result["input_timestamps"]
    pred_timestamps = result["prediction_timestamps"]
    frequency = result["config"]["frequency"]

    # 获取适合该频率的柱状图宽度
    bar_width = get_bar_width_for_frequency(frequency)

    plt.figure(figsize=(16, 10))

    # 计算要显示的历史数据量
    history_display_count = max(1, int(len(input_df) * history_display_ratio))
    recent_data = input_df.tail(history_display_count)
    recent_timestamps = input_timestamps.tail(history_display_count)

    # 绘制历史数据 - 使用细柱状图替代蜡烛图
    for i in range(len(recent_data)):
        timestamp = recent_timestamps.iloc[i]
        open_price = recent_data["open"].iloc[i]
        high_price = recent_data["high"].iloc[i]
        low_price = recent_data["low"].iloc[i]
        close_price = recent_data["close"].iloc[i]

        # 颜色设置
        color = "green" if close_price >= open_price else "red"

        # 绘制高低价细线
        plt.plot(
            [timestamp, timestamp],
            [low_price, high_price],
            color=color,
            linewidth=1,
            alpha=0.8,
        )

        # 绘制开盘收盘价细柱（关键改进：使用更细的柱状图）
        body_height = abs(close_price - open_price)
        if body_height > 0:
            bottom = min(open_price, close_price)
            plt.bar(
                timestamp,
                body_height,
                bottom=bottom,
                color=color,
                alpha=0.7,
                width=bar_width,
            )  # 使用自适应宽度

    # 绘制预测数据 - 同样使用细柱状图
    for i in range(len(pred_df)):
        timestamp = pred_timestamps.iloc[i]
        open_price = pred_df["open"].iloc[i]
        high_price = pred_df["high"].iloc[i]
        low_price = pred_df["low"].iloc[i]
        close_price = pred_df["close"].iloc[i]

        color = "lightgreen" if close_price >= open_price else "lightcoral"

        # 绘制高低价线
        plt.plot(
            [timestamp, timestamp],
            [low_price, high_price],
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )

        # 绘制开盘收盘价细柱（关键改进）
        body_height = abs(close_price - open_price)
        if body_height > 0:
            bottom = min(open_price, close_price)
            plt.bar(
                timestamp,
                body_height,
                bottom=bottom,
                color=color,
                alpha=0.8,
                width=bar_width,
            )  # 使用自适应宽度

    # 其余绘图代码保持不变...
    plt.axvline(
        x=input_timestamps.iloc[-1],
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Prediction Start",
    )

    # 设置标题等...
    avg_change = result["avg_change_ratio"]
    frequency_display = result["config"]["frequency"]
    num_predictions = result["config"]["num_predictions"]
    temperature = result["config"]["temperature"]
    top_p = result["config"]["top_p"]
    sample_count = result["config"]["sample_count"]
    plt.title(
        f"Kronos Adaptive Prediction ({frequency_display},t:{temperature},p:{top_p},{sample_count}x{num_predictions}) - Change: {avg_change:.2f}%",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Price", fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.xticks(rotation=30)
    plt.tight_layout()
    # 确保目标保存目录存在
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    plot_time = time.time() - plot_start_time
    print(f"📊 自适应图表已保存: {save_path} (绘图耗时: {plot_time:.2f}秒)")
    print(f"📈 图表优化: 使用{frequency}频率的自适应柱宽，解决蜡烛过宽问题")


def main():
    """
    主函数：执行预测流程
    """

    # 脚本开始时间
    script_start_time = time.time()
    start_datetime = datetime.now()

    print("🎯 优化预测策略")
    print("采样参数 - 3次直接平均（不去极值）")
    print("=" * 50)
    print(f"⏰ 脚本开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 配置参数
    config = {
        "data_file": "./data/XAUUSDM15_utf8.csv",  # 您的数据文件路径
        "model_name": "NeoQuasar/Kronos-small",  # 推荐从small开始
        "lookback": 512,  # 历史数据窗口
        "pred_len": 100,  # 预测未来120个周期
        "frequency": "M15",  # 数据频率，请匹配您的数据
        "num_predictions": 3,  # 预测次数（推荐2次以上）
        "temperature": 0.9,  # 金融预测用较低温度
        "top_p": 0.7,  # 适中的核采样
        "sample_count": 3,  # 每次多采样提高稳定性
    }

    print("📋 优化预测配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n🧠 参数优化原理:")
    print(f"  • 温度({config['temperature']}): 提高预测确定性，减少随机性")
    print(f"  • top_p({config['top_p']}): 平衡多样性与准确性")
    print(f"  • {config['sample_count']}次采样: 足够评估不确定性，避免过度计算")
    print(
        f"  • 预测{config['num_predictions']}次后平均: 避免损失有用信息，保持预测连续性"
    )
    print()

    # Ensure output directory relative to this script (examples/) instead of project root
    script_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 执行优化预测
    prediction_start_time = time.time()
    result = predict_multiple_and_average(**config)
    prediction_total_time = time.time() - prediction_start_time

    if result is not None:
        # 生成优化OHLC图表（保存到 output 目录）
        print("\n📊 生成优化OHLC预测图表...")
        plot_prediction_results_adaptive(
            result,
            save_path=os.path.join(
                output_dir, f"prediction_future_only_{config['frequency']}.png"
            ),
            history_display_ratio=0.10,  # 只显示部分的历史数据
            y_axis_expand_ratio=0.15,  # Y轴扩展15%
        )

        # 保存文件（全部写入 output 目录）
        print("\n💾 保存结果文件...")
        # 不导出 CSV，仅保留文本描述摘要保存逻辑
        save_start_time = time.time()
        print("💾 不导出 CSV，保留文本摘要保存逻辑...")

        # 将关键预测信息以文本形式保存到 output 目录
        try:
            pred_df = result["prediction"]
            pred_timestamps = result["prediction_timestamps"]
            last_price = result["last_known_price"]

            # 计算最高/最低及对应时间（基于平均预测结果）
            highs = pred_df["high"].values
            lows = pred_df["low"].values
            closes = pred_df["close"].values
            pos_max = int(np.argmax(highs))
            pos_min = int(np.argmin(lows))
            max_price = float(highs[pos_max])
            min_price = float(lows[pos_min])
            max_time = pd.to_datetime(pred_timestamps.iloc[pos_max])
            min_time = pd.to_datetime(pred_timestamps.iloc[pos_min])

            final_avg_price = float(closes[-1])
            change_percent = result.get(
                "avg_change_ratio", ((final_avg_price / last_price) - 1) * 100.0
            )
            direction = (
                "上涨"
                if final_avg_price > last_price
                else ("下跌" if final_avg_price < last_price else "持平")
            )

            text_summary_path = os.path.join(
                output_dir, f"prediction_future_only_{config['frequency']}.txt"
            )
            with open(text_summary_path, "w", encoding="utf-8") as f:
                f.write("Kronos 预测摘要\n")
                f.write("=================\n")
                f.write(f"最后已知价格: {last_price:.6f}\n")
                f.write(f"平均预测最终价格: {final_avg_price:.6f}\n")
                f.write(
                    f"平均预测相对最后已知价格变化: {change_percent:.4f}% ({direction})\n\n"
                )

                f.write("预测期内最高价及时间:\n")
                f.write(f"  最高价: {max_price:.6f}\n")
                f.write(f"  时间: {max_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("预测期内最低价及时间:\n")
                f.write(f"  最低价: {min_price:.6f}\n")
                f.write(f"  时间: {min_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("所有单次预测涨跌幅 (%):\n")
                for i, ratio in enumerate(result.get("change_ratios", [])):
                    f.write(f"  第{i+1}: {ratio:.4f}%\n")
                f.write("\n")

                f.write("计时信息 (秒):\n")
                timing = result.get("timing_info", {})
                for k in (
                    "model_load_time",
                    "predictor_init_time",
                    "data_load_time",
                    "data_prep_time",
                    "total_predictions_time",
                    "avg_prediction_time",
                    "analysis_time",
                ):
                    if k in timing:
                        f.write(f"  {k}: {timing[k]:.4f}\n")

                f.write("\n配置:\n")
                cfg = result.get("config", {})
                for k, v in cfg.items():
                    f.write(f"  {k}: {v}\n")

            save_time = time.time() - save_start_time
            print(
                f"💾 文本预测摘要已保存: {text_summary_path} (保存耗时: {save_time:.2f}秒)"
            )
        except Exception as e:
            save_time = time.time() - save_start_time
            print(f"⚠️ 无法保存文本摘要: {e} (耗时: {save_time:.2f}秒)")

        # 显示详细预测摘要
        pred_df = result["prediction"]
        pred_timestamps = result["prediction_timestamps"]
        last_price = result["last_known_price"]
        config_info = result["config"]
        timing_info = result["timing_info"]

        print("\n📈 优化预测摘要:")
        print(f"预测策略: 直接平均")
        print(f"预测次数: {config_info['num_predictions']} (全部使用)")
        print(f"有效预测: {len(result['all_predictions'])}")
        print(f"最后已知价格: {last_price:.4f}")
        print(f"预测时间跨度: {pred_timestamps.iloc[0]} 到 {pred_timestamps.iloc[-1]}")

        print(f"\n📊 所有预测结果涨跌幅:")
        for i, ratio in enumerate(result["change_ratios"]):
            print(f"  ✓ 第{i+1}次: {ratio:.2f}%")

        print(f"\n🎯 优化平均预测结果:")
        print(f"平均最终价格: {pred_df['close'].iloc[-1]:.4f}")
        print(f"平均总体变化: {result['avg_change_ratio']:.2f}%")
        print(f"平均预测最高价: {pred_df['high'].max():.4f}")
        print(f"平均预测最低价: {pred_df['low'].min():.4f}")

        # 显示预测稳定性指标
        change_ratios = result["change_ratios"]
        std_dev = np.std(change_ratios)
        print(f"\n📏 预测稳定性分析:")
        print(f"涨跌幅标准差: {std_dev:.2f}%")
        print(f"最大差异: {max(change_ratios) - min(change_ratios):.2f}%")
        print(f"变异系数: {(std_dev / np.mean(np.abs(change_ratios))) * 100:.1f}%")

        # 详细计时报告
        print(f"\n⏱️ 详细计时报告:")
        print(f"  模型加载时间: {timing_info['model_load_time']:.2f}秒")
        print(f"  预测器初始化时间: {timing_info['predictor_init_time']:.2f}秒")
        print(f"  数据加载时间: {timing_info['data_load_time']:.2f}秒")
        print(f"  数据准备时间: {timing_info['data_prep_time']:.2f}秒")
        print(f"  总预测时间: {timing_info['total_predictions_time']:.2f}秒")
        print(f"  平均单次预测时间: {timing_info['avg_prediction_time']:.2f}秒")
        print(f"  结果分析时间: {timing_info['analysis_time']:.2f}秒")
        print(f"  文件保存时间: {save_time:.2f}秒")
        print(f"  核心预测流程总时间: {prediction_total_time:.2f}秒")

        # 优化效果分析
        print(f"\n🚀 优化效果分析:")
        print(f"  • 低温度参数，提高预测确定性")
        print(f"  • 使用适中的核采样参数，平衡准确性与多样性")
        print(f"  • 3次直接平均，避免损失边缘信息")
        print(f"  • 每次预测内部多采样，提高单次预测质量")

    else:
        print("❌ 优化预测失败，请检查配置和数据")

    # 脚本结束时间和总耗时
    script_end_time = time.time()
    end_datetime = datetime.now()
    total_script_time = script_end_time - script_start_time

    print("\n" + "=" * 50)
    print("⏰ 脚本执行完成")
    print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总执行时间: {total_script_time:.2f}秒 ({total_script_time/60:.2f}分钟)")
    print("🎯 执行完毕！")
    print("=" * 50)


if __name__ == "__main__":
    main()
