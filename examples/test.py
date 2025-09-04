import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# 提取收盘价数据
sr_close = kline_df['close']
sr_pred_close = pred_df['close']

# 设置数据系列名称
sr_close.name = '历史数据'
sr_pred_close.name = "预测数据"

# 创建图表，设置大小为15x8英寸
fig, ax = plt.subplots(1, 1, figsize=(20, 6))

# 绘制历史数据
ax.plot(sr_close.index, sr_close.values, label='历史数据', color='blue', linewidth=1.5)

# 绘制预测数据
ax.plot(sr_pred_close.index, sr_pred_close.values, label='预测数据', color='red',
        linewidth=1.5, linestyle='--')

# 添加垂直线标记预测开始点
ax.axvline(x=sr_close.index[-1], color='green', linestyle=':', linewidth=1,
           label='预测起点')

# 设置图表标签和样式
ax.set_ylabel('收盘价格', fontsize=14)
ax.set_xlabel('日期', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_title('股价预测：未来20个交易日走势', fontsize=16)

# 设置X轴日期格式
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# 每2天显示一个主刻度
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))

# 旋转日期标签以避免重叠
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 调整布局以确保日期标签不被裁剪
plt.tight_layout()
plt.show()
