import math

# 第二问计算出的模型参数
beta_0 = -4.3909
beta_1 = 0.8528  # Fans
beta_2 = 0.6489  # GDP_pc
beta_3 = 0.1087  # Timezone
beta_4 = 0.0937  # Matches
beta_5 = 1.9602  # Comp

# 预设 2026 年中国市场的宏观与赛事参数
fans_cn = 300 
gdp_pc_cn = 13000 
timezone_cn = 0.6 
matches_cn = 104 
comp_cn = 0.7 

# 代入公式计算自然对数价格
ln_price = (beta_0 + 
            beta_1 * math.log(fans_cn) + 
            beta_2 * math.log(gdp_pc_cn) + 
            beta_3 * math.log(timezone_cn) + 
            beta_4 * math.log(matches_cn) + 
            beta_5 * math.log(comp_cn))

price_raw = math.exp(ln_price)
price_million = price_raw * 10        
price_billion = price_raw / 1000      

print("2026中国区理论转播权定价")
print(f"理论价格: {price_million:.2f} 万美元")
print(f"折合: {price_billion:.4f} 亿美元")