# ...existing code...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import time
import os
import shutil
from huggingface_hub import snapshot_download
import argparse

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
# ...existing code...

def infer_time_delta(ts_series):
    """使用历史中位时间差推断时间步长（返回 timedelta）"""
    diffs = ts_series.sort_values().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return timedelta(hours=1)
    median_seconds = int(diffs.median())
    if median_seconds <= 0:
        return timedelta(hours=1)
    return timedelta(seconds=median_seconds)


def make_future_timestamps(last_timestamp, step_td, pred_len):
    """生成未来 pred_len 个时间戳（基于 step_td）"""
    return pd.Series([last_timestamp + step_td * (i + 1) for i in range(pred_len)])


def plot_prediction_timeaxis(pred_out, out_png=None, time_label_fontsize=8, hour_interval=1):
    """按时间 X 轴绘制预测 close（保持不使用 volume），保存并显示 pyplot 窗口"""
    plt.figure(figsize=(10, 4))

    if 'timestamps' in pred_out.columns:
        x = pd.to_datetime(pred_out['timestamps'])
    else:
        x = pd.Series(range(len(pred_out)))

    y = pred_out['close'].values if 'close' in pred_out.columns else pred_out.iloc[:, -1].values

    ax = plt.gca()
    ax.plot(x, y, marker='o', linestyle='-', color='tab:orange', label='Prediction (close)')

    ax.set_title(f'Predicted Close (len={len(pred_out)})', fontsize=10)
    ax.set_xlabel('time', fontsize=9)
    ax.set_ylabel('price', fontsize=9)

    try:
        if pd.api.types.is_datetime64_any_dtype(x):
            # 每 hour_interval 小时为主刻度
            locator = mdates.HourLocator(interval=hour_interval)
            formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.gcf().autofmt_xdate(rotation=30)
            ax.tick_params(axis='x', labelsize=time_label_fontsize)
            ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)
    except Exception:
        ax.grid(True)

    ax.grid(True, which='both', axis='y', linestyle='-', alpha=0.2)
    ax.legend(fontsize=8)
    plt.tight_layout()

    # 如果指定输出图片路径则保存
    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot to: {out_png}")

    # 强制弹出 pyplot 窗口显示（按用户要求）
    try:
        plt.show()
    except Exception as e:
        print("Warning: plt.show() failed:", e)

    plt.close()


def ensure_ohlc_columns(df):
    required = ['open', 'high', 'low', 'close']
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少必需列: {c}")


def load_model_and_predictor(model_name, device, max_context=512, use_snapshot=False):
    if use_snapshot:
        # 可选：通过 snapshot_download 使用本地缓存（如果需要）
        try:
            snapshot_download(repo_id=model_name, local_dir=os.path.join(project_root, "model_cache"))
        except Exception:
            pass
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    return predictor


def normalize_pred_df(pred_df):
    """兼容 predictor 返回的多种格式，始终返回 DataFrame 且包含 close 列"""
    if isinstance(pred_df, pd.DataFrame):
        df = pred_df.copy()
    else:
        try:
            df = pd.DataFrame(pred_df)
        except Exception:
            raise SystemExit("predictor.predict 返回的结果无法转换为 DataFrame，请检查模型接口。")
    # 保证有 close/open/high/low 列，如果缺失则尝试填充或创建
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            df[col] = np.nan
    # 保持列顺序
    df = df[['open', 'high', 'low', 'close'] + [c for c in df.columns if c not in ['open','high','low','close']]]
    return df


def main():
    p = argparse.ArgumentParser(description="预测未来走势（不使用成交量）——增强版 prediction_wo_vol_fix")
    p.add_argument("--data", type=str, default="./data/XAUUSDM5_utf8.csv", help="输入 CSV 文件")
    p.add_argument("--lookback", type=int, default=512, help="上下文历史点数（<= predictor.max_context）")
    p.add_argument("--pred_len", type=int, default=100, help="预测步数")
    p.add_argument("--T", type=float, default=0.9, help="温度")
    p.add_argument("--top_p", type=float, default=0.7, help="top_p")
    p.add_argument("--sample_count", type=int, default=3, help="采样数量")
    p.add_argument("--device", type=str, default=None, help="设备，如 cpu 或 cuda:0")
    p.add_argument("--out_png", type=str, default="./examples/output/predicted_future_wo_vol.png", help="输出 PNG 路径")
    p.add_argument("--out_txt", type=str, default="./examples/output/predicted_future_wo_vol.txt", help="输出描述文本路径（不输出表格）")
    p.add_argument("--hour_interval", type=int, default=1, help="X 轴每隔多少小时画一条分割线")
    args = p.parse_args()

    # 选择设备
    device = args.device
    if device is None:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    start_time = time.time()

    # 加载模型/预测器（保持不使用 volume）
    predictor = load_model_and_predictor(model_name="NeoQuasar/Kronos-small", device=device, max_context=512)

    # 读取数据并校验
    df = pd.read_csv(args.data)
    if 'timestamps' not in df.columns:
        raise SystemExit("CSV 中缺少 timestamps 列。")
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    ensure_ohlc_columns(df)

    if args.lookback > len(df):
        raise SystemExit(f"lookback ({args.lookback}) 超过数据长度 ({len(df)})。")

    # 使用尾部历史作为上下文（保留不使用成交量列）
    x_df = df.tail(args.lookback)[['open', 'high', 'low', 'close']].reset_index(drop=True)
    x_timestamp = df.tail(args.lookback)['timestamps'].reset_index(drop=True)
    last_ts = x_timestamp.iloc[-1]

    # 生成未来时间戳（基于历史中位时间差）
    step_td = infer_time_delta(df['timestamps'])
    y_timestamp = make_future_timestamps(last_ts, step_td, args.pred_len)

    # 预测
    print("Running prediction...")
    t0 = time.time()
    pred_raw = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=args.T,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=True
    )
    t1 = time.time()
    print(f"Prediction time: {t1 - t0:.2f}s")

    pred_df = normalize_pred_df(pred_raw)

    # 将 timestamps 插入输出（但不保存表格），准备描述文本
    pred_out = pred_df.copy().reset_index(drop=True)
    pred_out.insert(0, 'timestamps', y_timestamp.values)

    # 生成描述文本（summary），写入 out_txt
    os.makedirs(os.path.dirname(args.out_txt) or '.', exist_ok=True)
    try:
        last_known_price = float(x_df['close'].iloc[-1])
    except Exception:
        last_known_price = float(np.nan)

    try:
        first_pred = float(pred_out['close'].iloc[0])
        last_pred = float(pred_out['close'].iloc[-1])
        pct_change = (last_pred - last_known_price) / last_known_price * 100 if last_known_price and not np.isnan(last_known_price) else float('nan')
    except Exception:
        first_pred = last_pred = pct_change = float('nan')

    summary_lines = [
        f"model: NeoQuasar/Kronos-small",
        f"device: {device}",
        f"lookback: {args.lookback}",
        f"pred_len: {args.pred_len}",
        f"T: {args.T}",
        f"top_p: {args.top_p}",
        f"sample_count: {args.sample_count}",
        f"last_known_timestamp: {last_ts}",
        f"last_known_price: {last_known_price}",
        f"first_pred_close: {first_pred}",
        f"last_pred_close: {last_pred}",
        f"predicted_total_pct_change: {pct_change:.4f}%",
        f"prediction_timestamp_start: {pred_out['timestamps'].iloc[0]}",
        f"prediction_timestamp_end: {pred_out['timestamps'].iloc[-1]}",
        f"prediction_time_seconds: {t1 - t0:.4f}",
        f"total_elapsed_seconds: {time.time() - start_time:.4f}"
    ]

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved prediction summary to: {args.out_txt}")

    # 绘图（时间轴，小时分割线，较小时间标签字号）并显示/保存
    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)
    plot_prediction_timeaxis(pred_out, out_png=args.out_png, time_label_fontsize=8, hour_interval=args.hour_interval)

    elapsed = time.time() - start_time
    print(f"Total elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()