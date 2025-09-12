# =======================
# 文件名: future_prediction_optimized.py
# 基于Kronos推荐的采样参数优化版本
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
    compare_target_len,
):
    """
    基于推荐参数执行多次预测并选择最接近真实走势的预测（若数据包含用于比较的片段）。
    修改：使用导入数据的最后 50 条记录作为对比片段（若可用），只用预测的前 eval_len 步与该片段比较。
    若不可用则退回到原行为（使用文件尾作为历史上下文，无比较）。

    返回:
    dict, 包含最终选定的预测（'prediction'），以及所有候选预测、评估信息等
    """
    print("🚀 基于Kronos优化的预测策略")
    print("=" * 50)
    print(f"预测次数: {num_predictions} (不去极值，后续根据可用真实数据选择或平均)")
    print(f"温度参数: {temperature} (推荐用于金融预测)")
    print(f"核采样参数: {top_p}")
    print(f"单次采样数: {sample_count}")
    print()

    # 步骤1: 加载模型和分词器（只需加载一次）
    print("📥 加载预训练模型...")
    model_start_time = time.time()

    try:
        # 原有缓存 / 下载逻辑（保持不变）
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        backup_cache = os.path.join(project_root, "model_cache_backup")
        z_root_exists = os.path.exists("Z:\\") and os.path.isdir("Z:\\")
        z_cache = r"Z:\model_cache"

        proxy = "http://127.0.0.1:11082"
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("http_proxy", proxy)
        os.environ.setdefault("https_proxy", proxy)

        cache_dir_to_use = None

        if z_root_exists and os.path.exists(z_cache) and len(os.listdir(z_cache)) > 0:
            cache_dir_to_use = z_cache
            print(f"⚡ 使用 Z: 盘缓存 -> {z_cache} (无需覆盖)")
        elif os.path.exists(backup_cache) and len(os.listdir(backup_cache)) > 0:
            cache_dir_to_use = backup_cache
            print(f"📁 使用项目内持久缓存 -> {backup_cache} (无需下载)")
        else:
            download_target = z_cache if z_root_exists else backup_cache
            os.makedirs(download_target, exist_ok=True)
            print(f"📥 开始下载至 -> {download_target} ...")
            local_tokenizer_dir = snapshot_download(
                repo_id="NeoQuasar/Kronos-Tokenizer-base",
                cache_dir=download_target,
                resume_download=True,
            )
            local_model_dir = snapshot_download(
                repo_id=model_name, cache_dir=download_target, resume_download=True
            )
            try:
                if download_target == z_cache:
                    if (
                        not os.path.exists(backup_cache)
                        or len(os.listdir(backup_cache)) == 0
                    ):
                        if os.path.exists(backup_cache):
                            try:
                                shutil.rmtree(backup_cache)
                            except Exception:
                                pass
                        shutil.copytree(z_cache, backup_cache)
                        print(f"💾 已将 Z: 缓存备份到项目目录 -> {backup_cache}")
                cache_dir_to_use = download_target
            except Exception:
                cache_dir_to_use = download_target

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

    # 步骤4: 准备输入数据（使用导入数据最后 50 条作为比较目标）
    print("🎯 准备预测输入...")
    prep_start_time = time.time()

    total_length = len(df)
    # compare_target_len 由调用方通过 config 提供；确保不超过可用数据
    compare_target_len = min(compare_target_len, max(0, total_length - 1))
    # 若无法构造比较目标，则回退到原行为（使用文件末尾作为历史）
    if compare_target_len > 0 and total_length - compare_target_len - 1 >= 0:
        # 使用最后 compare_target_len 条作为 ground truth 比较段
        history_end_idx = total_length - compare_target_len
        start_idx = max(0, history_end_idx - lookback)
        end_idx = history_end_idx
        eval_len = min(compare_target_len, pred_len)
        print(
            f"使用最后 {compare_target_len} 条作为比较目标，eval_len={eval_len}, history_end_idx={history_end_idx}"
        )
        ground_truth_df = df.iloc[
            history_end_idx : history_end_idx + compare_target_len
        ].reset_index(drop=True)
    else:
        # 回退：无比较段
        start_idx = max(0, total_length - lookback)
        end_idx = total_length
        history_end_idx = end_idx
        eval_len = 0
        ground_truth_df = None
        print("无法使用最后50条作为比较目标，采用文件末尾作为已知历史（无比较段）")

    if "volume" in df.columns and "amount" in df.columns:
        x_df = df.iloc[start_idx:end_idx][
            ["open", "high", "low", "close", "volume", "amount"]
        ].reset_index(drop=True)
        print("✓ 使用完整的OHLCVA数据")
    else:
        x_df = df.iloc[start_idx:end_idx][["open", "high", "low", "close"]].reset_index(
            drop=True
        )
        print("✓ 使用OHLC基础数据")

    x_timestamp = df.iloc[start_idx:end_idx]["timestamps"].reset_index(drop=True)
    last_known_time = df["timestamps"].iloc[end_idx - 1]
    last_known_price = float(df["close"].iloc[end_idx - 1])
    # 预测时间戳（从 last_known_time 开始）
    y_timestamp = generate_future_timestamps(last_known_time, pred_len, frequency)

    prep_time = time.time() - prep_start_time
    print(f"输入数据范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
    if ground_truth_df is not None:
        print(
            f"用于验证的真实比较段: {ground_truth_df['timestamps'].iloc[0]} 到 {ground_truth_df['timestamps'].iloc[-1]}"
        )
    print(f"✅ 数据准备完成 (耗时: {prep_time:.2f}秒)")

    # 步骤5: 执行多次预测并保存候选结果
    print(f"🔮 开始执行{num_predictions}次预测（使用推荐参数）...）")
    predictions_start_time = time.time()

    candidate_predictions = []
    candidate_change_ratios = []
    candidate_times = []

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

            single_pred_time = time.time() - single_pred_start
            candidate_times.append(single_pred_time)

            # 计算相对于最后已知价格的涨跌幅（针对完整预测）
            change_ratio = calculate_price_change_ratio(pred_df, last_known_price)

            candidate_predictions.append(pred_df.copy().reset_index(drop=True))
            candidate_change_ratios.append(change_ratio)

            print(f"完成 (耗时: {single_pred_time:.2f}秒, 涨跌幅: {change_ratio:.2f}%)")

        except Exception as e:
            single_pred_time = time.time() - single_pred_start
            print(f"失败 (耗时: {single_pred_time:.2f}秒) - {e}")
            continue

    total_predictions_time = time.time() - predictions_start_time

    if len(candidate_predictions) < 1:
        print(f"❌ 预测全部失败")
        return None

    print(f"✅ 完成{len(candidate_predictions)}次有效预测")
    print(
        f"📊 预测统计: 总预测时间 {total_predictions_time:.2f}s, 平均单次 {np.mean(candidate_times):.2f}s"
    )

    # 步骤6: 若存在比较目标（ground_truth_df）则比较并选择最贴近实际走势的预测的前 eval_len 步；否则对候选取平均
    best_idx = None
    best_pred_df = None
    evaluation_metrics = {}

    if ground_truth_df is not None and eval_len > 0:
        # 使用均方误差(MSE)在前 eval_len 个 close 上比较
        gt_close = ground_truth_df["close"].values.astype(float)[:eval_len]
        errors = []
        for idx, cand in enumerate(candidate_predictions):
            pred_close = cand["close"].values.astype(float)[:eval_len]
            if len(pred_close) < eval_len:
                pred_close = np.pad(
                    pred_close, (0, eval_len - len(pred_close)), constant_values=np.nan
                )
            mask = ~np.isnan(pred_close)
            if mask.sum() == 0:
                mse = float("inf")
            else:
                mse = float(np.mean((pred_close[mask] - gt_close[mask]) ** 2))
            errors.append(mse)

        best_idx = int(np.nanargmin(errors))
        best_pred_df = candidate_predictions[best_idx]
        evaluation_metrics["mse_list"] = errors
        evaluation_metrics["best_idx"] = best_idx
        evaluation_metrics["best_mse"] = float(errors[best_idx])
        print(
            f"🔎 已比较 {len(candidate_predictions)} 个候选（对比前 {eval_len} 步），选择最接近的预测: idx={best_idx}, mse={errors[best_idx]:.6f}"
        )

    else:
        # 无比较段 -> 对候选预测逐点平均
        print("ℹ️ 未检测到可比较的真实段，采用候选预测逐点平均作为最终结果")
        avg_pred_df = candidate_predictions[0].copy()
        numeric_columns = ["open", "high", "low", "close"]
        if "volume" in avg_pred_df.columns:
            numeric_columns.append("volume")
        if "amount" in avg_pred_df.columns:
            numeric_columns.append("amount")
        for col in numeric_columns:
            avg_pred_df[col] = 0.0
            for cand in candidate_predictions:
                avg_pred_df[col] += cand[col].astype(float)
            avg_pred_df[col] = avg_pred_df[col] / len(candidate_predictions)
        best_pred_df = avg_pred_df.reset_index(drop=True)
        best_idx = None
        evaluation_metrics["mse_list"] = None
        evaluation_metrics["best_idx"] = None
        evaluation_metrics["best_mse"] = None

    analysis_time = time.time() - prep_start_time
    print(f"✅ 最终预测选择/合成完成 (耗时: {analysis_time:.2f}s)")

    # 计算平均变化率以便上层展示
    try:
        avg_change_ratio = (
            float(np.mean(candidate_change_ratios))
            if len(candidate_change_ratios) > 0
            else 0.0
        )
    except Exception:
        avg_change_ratio = 0.0

    # 统一返回 key 为 'prediction' 的 DataFrame 以便下游兼容
    return {
        "prediction": best_pred_df,
        "best_index": best_idx,
        "candidate_predictions": candidate_predictions,
        "all_predictions": candidate_predictions,  # 兼容旧键名
        "change_ratios": candidate_change_ratios,
        "avg_change_ratio": avg_change_ratio,
        "prediction_timestamps": y_timestamp,
        "last_known_price": last_known_price,
        "ground_truth": ground_truth_df,
        "evaluation": evaluation_metrics,
        "timing_info": {
            "model_load_time": model_load_time,
            "predictor_init_time": predictor_init_time,
            "data_load_time": data_load_time,
            "data_prep_time": prep_time,
            "total_predictions_time": total_predictions_time,
            "avg_prediction_time": (
                np.mean(candidate_times) if len(candidate_times) > 0 else 0.0
            ),
            "analysis_time": analysis_time,
            "prediction_times": candidate_times,
        },
        "config": {
            "model_name": model_name,
            "lookback": lookback,
            "pred_len": pred_len,
            "frequency": frequency,
            "num_predictions": num_predictions,
            "used_predictions": len(candidate_predictions),
            "temperature": temperature,
            "top_p": top_p,
            "sample_count": sample_count,
        },
        # 为绘图与外部逻辑提供输入数据与时间戳
        "input_data": x_df,
        "input_timestamps": x_timestamp,
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
    """自适应绘制预测结果 - 同时展示用于比较的真实未来片段（若有）、候选预测与最终选定预测"""
    if result is None:
        print("❌ 无预测结果可绘制")
        return

    print("📊 开始生成自适应图表（包含比较段与预测段）...")
    plot_start_time = time.time()

    pred_df = result["prediction"]
    input_df = result["input_data"]
    input_timestamps = pd.to_datetime(result["input_timestamps"]).reset_index(drop=True)
    pred_timestamps = pd.to_datetime(result["prediction_timestamps"]).reset_index(
        drop=True
    )
    frequency = result["config"].get("frequency", "H1")

    # 可选内容
    ground_truth = result.get("ground_truth", None)
    candidate_predictions = result.get("candidate_predictions", [])
    best_index = result.get("best_index", None)

    # 获取适合该频率的柱状图宽度（若需要OHLC柱）
    bar_width = get_bar_width_for_frequency(frequency)

    plt.figure(figsize=(16, 9))

    # 绘制最近历史数据（按比例）
    history_display_count = max(1, int(len(input_df) * history_display_ratio))
    recent_data = input_df.tail(history_display_count).reset_index(drop=True)
    recent_timestamps = input_timestamps.tail(history_display_count).reset_index(
        drop=True
    )

    # 历史：用细线绘制 close，并用细柱表示OHLC（保留原视觉）
    plt.plot(
        recent_timestamps,
        recent_data["close"].values,
        color="black",
        linewidth=1.2,
        label="History (close)",
    )

    # 如果存在真实未来片段，绘制在历史之后的比较段（实线）
    if ground_truth is not None and len(ground_truth) > 0:
        gt_ts = pd.to_datetime(ground_truth["timestamps"]).reset_index(drop=True)
        gt_close = ground_truth["close"].astype(float).values
        plt.plot(
            gt_ts, gt_close, color="k", linewidth=2.0, label="Ground Truth (close)"
        )
    else:
        gt_ts = None

    # 绘制所有候选预测（半透明虚线），用于显示不确定性
    for idx, cand in enumerate(candidate_predictions):
        try:
            cand_ts = pred_timestamps.iloc[: len(cand)]
            plt.plot(
                cand_ts,
                cand["close"].astype(float).values,
                color="gray",
                linestyle="--",
                alpha=0.25,
                linewidth=1,
                label="Candidate Predictions" if idx == 0 else None,
            )
        except Exception:
            continue

    # 绘制最终选定的预测（或平均预测），用醒目颜色
    try:
        final_ts = pred_timestamps.iloc[: len(pred_df)]
        plt.plot(
            final_ts,
            pred_df["close"].astype(float).values,
            color="tab:blue",
            linewidth=2.2,
            label="Selected Prediction",
        )
        # 标注最终预测的最高与最低点
        max_pos = int(np.argmax(pred_df["high"].values))
        min_pos = int(np.argmin(pred_df["low"].values))
        plt.scatter(
            [final_ts.iloc[max_pos]],
            [pred_df["high"].iloc[max_pos]],
            color="tab:green",
            s=50,
            zorder=5,
            label=(
                "Predicted High"
                if "Predicted High" not in plt.gca().get_legend_handles_labels()[1]
                else None
            ),
        )
        plt.scatter(
            [final_ts.iloc[min_pos]],
            [pred_df["low"].iloc[min_pos]],
            color="tab:red",
            s=50,
            zorder=5,
            label=(
                "Predicted Low"
                if "Predicted Low" not in plt.gca().get_legend_handles_labels()[1]
                else None
            ),
        )
    except Exception:
        pass

    # 若存在 ground truth 并且其区间与预测时间有重叠，绘制残差（可选，绘在右侧小窗或作为点）
    if ground_truth is not None and len(ground_truth) > 0:
        # 对比段：若长度一致，则绘制误差线（灰色细棒）
        try:
            compare_len = min(len(ground_truth), len(pred_df))
            comp_ts = pd.to_datetime(ground_truth["timestamps"]).reset_index(drop=True)[
                :compare_len
            ]
            error = (
                pred_df["close"].astype(float).values[:compare_len]
                - ground_truth["close"].astype(float).values[:compare_len]
            )
            # 在主图上绘制误差的散点（以辅助视觉显示误差方向）
            err_colors = ["tab:green" if v >= 0 else "tab:red" for v in error]
            plt.scatter(
                comp_ts,
                ground_truth["close"].astype(float).values[:compare_len],
                c=err_colors,
                s=20,
                alpha=0.9,
                marker="x",
                label="GT (+) / (-) err" if True else None,
            )
        except Exception:
            pass

    # 视觉与格式化
    plt.axvline(
        x=input_timestamps.iloc[-1],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Prediction Start",
    )

    avg_change = result.get("avg_change_ratio", 0.0)
    freq_display = frequency
    num_predictions = result["config"]["num_predictions"]
    temperature = result["config"]["temperature"]
    top_p = result["config"]["top_p"]
    sample_count = result["config"]["sample_count"]
    plt.title(
        f"Kronos Prediction Comparison ({freq_display},t:{temperature},p:{top_p},{sample_count}x{num_predictions}) - Change: {avg_change:.2f}%",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Price", fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.xticks(rotation=30)
    plt.legend(fontsize=10, loc="best")

    plt.tight_layout()

    # 保存
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    plot_time = time.time() - plot_start_time
    print(f"📊 自适应对比图已保存: {save_path} (耗时: {plot_time:.2f}s)")


def main():
    """
    主函数：执行基于推荐的优化预测流程
    """

    # 脚本开始时间
    script_start_time = time.time()
    start_datetime = datetime.now()

    print("🎯 Kronos优化预测策略")
    print("基于推荐采样参数 - 3次直接平均（不去极值）")
    print("=" * 50)
    print(f"⏰ 脚本开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 配置参数（基于推荐）
    config = {
        "data_file": "./data/XAUUSDM15_utf8.csv",  # 您的数据文件路径
        "model_name": "NeoQuasar/Kronos-small",  # 推荐从small开始
        "lookback": 512,  # 历史数据窗口
        "frequency": "M15",  # 数据频率，请匹配您的数据
        "num_predictions": 3,  # 增加候选次数（建议 5-10）
        "temperature": 0.9,  # 降低温度提高确定性
        "top_p": 0.7,  # 略收紧核采样
        "sample_count": 3,  # 每次内部采样增大以提升单次质量
        "compare_target_len": 20,  # 用于比较的真实数据条数（可调，建议 20-100）
        "pred_len": 100,  # 预测未来的周期个数
    }

    print("📋 优化预测配置（基于推荐）:")
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
                output_dir, f"prediction_future_compare_{config['frequency']}.png"
            ),
            history_display_ratio=0.10,  # 只显示部分的历史数据
            y_axis_expand_ratio=0.15,  # Y轴扩展15%
        )

        # 保存文件（全部写入 output 目录）
        print("\n💾 保存结果文件...")
        save_start_time = time.time()

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
                output_dir, f"prediction_future_compare_{config['frequency']}.txt"
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
        print(f"预测策略: 推荐参数 + 直接平均")
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
        print(f"  • 采用推荐的低温度参数，提高预测确定性")
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
    print("🎯 基于Kronos的优化策略执行完毕！")
    print("=" * 50)


if __name__ == "__main__":
    main()
