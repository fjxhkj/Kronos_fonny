"""
直接预测未来走势（不覆盖原 examples/prediction_example.py）

用途：
 - 从历史数据末尾取 lookback 条作为上下文，预测接下来的 pred_len 步
 - 将预测结果保存为 CSV，并输出预测收盘价图（PNG）

用法（在项目根运行）：
  python .\examples\prediction_future_only.py --data ./data/XAUUSDH1_utf8.csv --lookback 512 --pred_len 256
"""
import argparse
import os
import sys
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def infer_time_delta(ts_series):
    """使用中位时间差作为步长（返回 timedelta）"""
    diffs = ts_series.sort_values().diff().dropna().dt.total_seconds()
    if len(diffs) == 0:
        return timedelta(hours=1)
    median_seconds = int(diffs.median())
    if median_seconds <= 0:
        return timedelta(hours=1)
    return timedelta(seconds=median_seconds)


def make_future_timestamps(last_timestamp, step_td, pred_len):
    """生成未来 pred_len 个时间戳"""
    return pd.Series([last_timestamp + step_td * (i + 1) for i in range(pred_len)])


def plot_pred_only(pred_df, out_png=None):
    """仅绘制预测的 close 曲线并按时间显示 X 轴（小时刻度线），并减小时间标签字号"""
    plt.figure(figsize=(10, 4))

    # 若有 timestamps 列则使用具体时间作为 X 轴
    if 'timestamps' in pred_df.columns:
        x = pd.to_datetime(pred_df['timestamps']).reset_index(drop=True)
    else:
        # 回退到索引
        x = pd.Series(range(len(pred_df)))

    y = pred_df['close'].values if 'close' in pred_df.columns else pred_df.iloc[:, -1].values

    ax = plt.gca()
    ax.plot(x, y, marker='o', linestyle='-', color='tab:orange', label='Predicted Close')

    # 设置较小字号，避免时间文字过大
    ax.set_title(f'Predicted Close (len={len(pred_df)})', fontsize=10)
    ax.set_xlabel('time', fontsize=9)
    ax.set_ylabel('price', fontsize=9)

    # 如果 X 轴为时间类型，则使用每小时主刻度并显示竖直分割线
    try:
        if pd.api.types.is_datetime64_any_dtype(x):
            locator = mdates.HourLocator(interval=1)
            formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            # 使标签倾斜便于阅读并设置较小字号
            plt.gcf().autofmt_xdate(rotation=30)
            ax.tick_params(axis='x', labelsize=8)  # 关键：缩小 X 轴刻度文字
            # 仅在 X 轴显示主刻度的网格线（竖线）
            ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)
    except Exception:
        # 若发生任何格式化错误，退回到简单网格
        ax.grid(True)

    # 常规网格与图例（图例字号也调小）
    ax.grid(True, which='both', axis='y', linestyle='-', alpha=0.2)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot to: {out_png}")
    else:
        plt.show()
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/XAUUSDH1_utf8.csv", help="输入 CSV 文件，需包含 timestamps, open, high, low, close, volume, amount")
    p.add_argument("--lookback", type=int, default=512, help="用于上下文的历史点数（不能超过模型 max_context）")
    p.add_argument("--pred_len", type=int, default=256, help="要预测的未来步数")
    p.add_argument("--T", type=float, default=1.0, help="温度")
    p.add_argument("--top_p", type=float, default=0.9, help="top_p")
    p.add_argument("--sample_count", type=int, default=1, help="每次采样数量")
    p.add_argument("--device", type=str, default=None, help="设备，比如 cpu 或 cuda:0（默认自动选择）")
    p.add_argument("--out_csv", type=str, default="./examples/predicted_future.csv", help="输出预测 CSV 路径")
    p.add_argument("--out_png", type=str, default="./examples/predicted_future.png", help="输出预测图片路径")
    args = p.parse_args()

    # 加载 tokenizer 与 model
    print("Loading tokenizer and model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    # 选择设备
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"Using device: {device}")

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # 读取数据并准备上下文（取最后 lookback 条）
    df = pd.read_csv(args.data)
    if 'timestamps' not in df.columns:
        raise SystemExit("CSV 中缺少 timestamps 列。")
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    if args.lookback > len(df):
        raise SystemExit(f"lookback ({args.lookback}) 超过数据长度 ({len(df)})。")

    # 保证必要列存在
    required_cols = ['open', 'high', 'low', 'close']
    for c in required_cols:
        if c not in df.columns:
            raise SystemExit(f"CSV 缺少必需列: {c}")

    # volume/amount 如果缺失则填充 0
    if 'volume' not in df.columns:
        df['volume'] = 0
    if 'amount' not in df.columns:
        df['amount'] = 0

    x_df = df.tail(args.lookback)[['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
    x_timestamp = df.tail(args.lookback)['timestamps'].reset_index(drop=True)
    last_ts = x_timestamp.iloc[-1]

    # 生成未来时间戳（基于历史中位时间差）
    step_td = infer_time_delta(df['timestamps'])
    y_timestamp = make_future_timestamps(last_ts, step_td, args.pred_len)

    # 进行预测
    print("Running prediction...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=args.T,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=True
    )

    # 检查返回值并保存
    if not isinstance(pred_df, pd.DataFrame):
        # 若 predictor 返回 dict 或其他格式，尽量转换
        try:
            pred_df = pd.DataFrame(pred_df)
        except Exception:
            raise SystemExit("predictor.predict 返回的结果无法转换为 DataFrame，请检查模型接口。")

    # 补齐 timestamps 并保存
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    pred_out = pred_df.copy()
    pred_out.insert(0, 'timestamps', y_timestamp.values)
    pred_out.to_csv(args.out_csv, index=False, encoding='utf-8')
    print(f"Saved predictions to: {args.out_csv}")

    # 可视化预测 close（仅未来部分）
    if 'close' in pred_out.columns:
        plot_pred_only(pred_out, out_png=args.out_png)
    else:
        print("预测结果中不包含 'close' 列，跳过绘图。")


if __name__ == "__main__":
    main()