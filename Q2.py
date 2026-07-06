import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

# 1. 修正后的数据集：
data_refined = [
    # 2018
    ['英国', 2018, 1.07, 29.93, 4.36, 1.2, 64, 1.0],
    ['日本', 2018, 1.36, 31.63, 3.91, 0.8, 64, 1.3],
    ['韩国', 2018, 0.55, 21.16, 3.33, 0.8, 64, 1.2],
    ['德国', 2018, 1.24, 33.99, 4.79, 1.2, 64, 1.2],
    ['越南', 2018, 0.12, 74.49, 0.32, 0.8, 64, 0.7],
    ['巴西', 2018, 2.00, 125.64, 0.92, 1.0, 64, 1.0],
    ['澳大利亚', 2018, 0.15, 6.97, 5.74, 0.7, 64, 1.0],
    ['中国香港', 2018, 0.26, 4.13, 4.80, 0.8, 64, 0.9],
    ['中国', 2018, 1.50, 306.99, 1.00, 0.8, 64, 0.7],
    # 2022
    ['英国', 2022, 0.99, 30.15, 4.63, 1.2, 64, 1.0],
    ['日本', 2022, 1.14, 32.53, 3.41, 0.9, 64, 1.3],
    ['韩国', 2022, 0.55, 20.68, 3.23, 0.9, 64, 1.2],
    ['德国', 2022, 1.11, 35.20, 4.87, 1.2, 64, 1.2],
    ['越南', 2022, 0.15, 78.56, 0.42, 0.9, 64, 0.7],
    ['巴西', 2022, 2.20, 133.49, 0.91, 1.0, 64, 1.0],
    ['澳大利亚', 2022, 0.14, 7.80, 6.50, 0.8, 64, 1.0],
    ['中国香港', 2022, 0.38, 4.07, 4.86, 0.9, 64, 0.9],
    ['中国', 2022, 1.50, 338.88, 1.27, 0.9, 64, 0.7],
    # 2026
    ['英国', 2026, 1.75, 31.28, 5.44, 0.6, 104, 1.0],
    ['日本', 2026, 2.00, 34.16, 3.57, 0.5, 104, 1.3],
    ['韩国', 2026, 1.25, 21.84, 3.46, 0.5, 104, 1.2],
    ['德国', 2026, 1.40, 36.96, 5.60, 0.6, 104, 1.2],
    ['越南', 2026, 0.15, 83.83, 0.50, 0.5, 104, 0.7],
    ['巴西', 2026, 1.10, 141.70, 1.03, 1.2, 104, 1.0],
    ['澳大利亚', 2026, 0.15, 8.91, 6.67, 0.6, 104, 1.0],
    ['中国香港', 2026, 0.25, 4.41, 5.13, 0.5, 104, 0.9]
]

cols = ['Country', 'Year', 'Price', 'Fans', 'GDP_pc', 'Timezone', 'Matches', 'Comp']
df = pd.DataFrame(data_refined, columns=cols)

# 2. 对数化处理
df['ln_Price'] = np.log(df['Price'])
df['ln_Fans'] = np.log(df['Fans'])
df['ln_GDP_pc'] = np.log(df['GDP_pc'])
df['ln_Timezone'] = np.log(df['Timezone'])
df['ln_Matches'] = np.log(df['Matches'])
df['ln_Comp'] = np.log(df['Comp'])

# 3. 构建 OLS 回归
X = df[['ln_Fans', 'ln_GDP_pc', 'ln_Timezone', 'ln_Matches', 'ln_Comp']]
X = sm.add_constant(X) 
Y = df['ln_Price']
model = sm.OLS(Y, X).fit()

# 打印结果
print("模型参数：")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Intercept(beta_0): {model.params['const']:.4f}")
print(f"Fans(beta_1): {model.params['ln_Fans']:.4f}")
print(f"GDP_pc(beta_2): {model.params['ln_GDP_pc']:.4f}")
print(f"Timezone(beta_3): {model.params['ln_Timezone']:.4f}")
print(f"Matches(beta_4): {model.params['ln_Matches']:.4f}")
print(f"Comp(beta_5): {model.params['ln_Comp']:.4f}")

# 4. 生成的预测图
df['Predicted_ln_Price'] = model.predict(X)
df['Predicted_Price'] = np.exp(df['Predicted_ln_Price'])

plt.figure(figsize=(9, 6))
sns.scatterplot(x='Price', y='Predicted_Price', hue='Country', data=df, s=120, palette='tab10', edgecolor='black', alpha=0.9)

max_val = max(df['Price'].max(), df['Predicted_Price'].max())
plt.plot([0, max_val + 0.2], [0, max_val + 0.2], 'r--', lw=2, label='完美预测线 (y=x)')

plt.title('定价模型检验', fontsize=15, fontweight='bold')
plt.xlabel('实际公开报道成交价 (亿美元)', fontsize=12, fontweight='bold')
plt.ylabel('模型理论预测价 (亿美元)', fontsize=12, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 5. 生成的弹性系数图
elasticities = {
    '绝对球迷基数 (Fans)': model.params['ln_Fans'],
    '人均GDP消费力 (GDP_pc)': model.params['ln_GDP_pc'],
    '买方竞争格局 (Comp)': model.params['ln_Comp'],
    '比赛扩容场次 (Matches)': model.params['ln_Matches'],
    '收视时差重合度 (Timezone)': model.params['ln_Timezone']
}

coef_df = pd.DataFrame(list(elasticities.items()), columns=['Factor', 'Elasticity']).sort_values(by='Elasticity', ascending=True)

plt.figure(figsize=(9, 5))
bars = plt.barh(coef_df['Factor'], coef_df['Elasticity'], color='#2ecc71')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)

plt.title('全球定价模型：因子价值弹性分析', fontsize=15, fontweight='bold')
plt.xlabel('价值弹性系数', fontsize=12, fontweight='bold')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', ha='left', fontweight='bold')

plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()