import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# 字体与全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

# 数据输入
data_jpn = {
    'Year': ['2018(俄罗斯)', '2022(卡塔尔)', '2026(美加墨)'],
    'Price': [1.36, 1.14, 2.00],          # 成交价(亿美元)
    'Matches': [64, 64, 104],             # 比赛容量
    'GDP_pc': [3.91, 3.41, 3.57]          # 人均GDP(万美元)
}
df = pd.DataFrame(data_jpn)

# 图1：日本历届转播费与宏观经济波动
fig, ax1 = plt.subplots(figsize=(9, 5.5))

# 柱状图：转播费
bars = ax1.bar(df['Year'], df['Price'], color=['#95a5a6', '#7f8c8d', '#2c3e50'], width=0.4, label='转播权成交价 (亿美元)')
ax1.set_ylabel('成交价 (亿美元)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 2.5)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.03, f"{yval}亿", ha='center', va='bottom', fontsize=12, fontweight='bold')

# 折线图：人均GDP
ax2 = ax1.twinx()
line = ax2.plot(df['Year'], df['GDP_pc'], color='#f39c12', marker='D', linewidth=2.5, markersize=8, label='人均GDP (万美元)')
ax2.set_ylabel('人均GDP (万美元)', fontsize=12, fontweight='bold', color='#d35400')
ax2.set_ylim(2.5, 4.5)
ax2.tick_params(axis='y', labelcolor='#d35400')

plt.title('图1：日本转播费回落之谜——受宏观经济(人均GDP)波动的严格调节', fontsize=14, fontweight='bold')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('图1日本历届趋势与经济变化.png', dpi=300) 
plt.show() 
plt.close()


# 图2：估价模型参数反推与博弈分析
P_2022 = 1.14
P_2026 = 2.00
V_ratio = 104 / 64          # 容量扩军率 = 1.625
Z_ratio = 0.5 / 0.9         # 时差恶化率 = 0.555
M_market = (34.16 / 32.53) * (3.57 / 3.41)  # 综合宏观增量 = 1.100

target_val = P_2026 / (P_2022 * M_market) 

alphas = np.linspace(0.8, 3.0, 100) 

def calculate_beta(alpha, C_comp):
    return (np.log(target_val) - np.log(C_comp) - alpha * np.log(V_ratio)) / np.log(Z_ratio)

betas_normal = calculate_beta(alphas, 1.0)
betas_premium = calculate_beta(alphas, 1.2)

plt.figure(figsize=(9, 6))
plt.plot(alphas, betas_normal, 'b-', linewidth=2.5, alpha=0.6, label='场景A：维持常规竞争强度')
plt.plot(alphas, betas_premium, 'r-', linewidth=2.5, label='场景B：流媒体加入')

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.8)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.8)

plt.fill_between(alphas, -0.5, 0.5, where=(alphas >= 1.0) & (alphas <= 2.2), 
                 color='green', alpha=0.15, 
                 label='多媒体/流媒体联合购买的可行区间\n')

plt.xlabel('赛事扩军价值弹性 (α)', fontsize=12, fontweight='bold')
plt.ylabel('时差折损敏感度 (β)', fontsize=12, fontweight='bold')
plt.title('图2：日本2026转播权估价模型：参数博弈与流媒体效应', fontsize=14, fontweight='bold')
plt.ylim(-1.5, 2.5)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

plt.savefig('图2日本博弈分析.png', dpi=300)
plt.show()  
plt.close()