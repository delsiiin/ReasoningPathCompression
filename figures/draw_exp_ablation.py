import numpy as np
import matplotlib.pyplot as plt

# ========== 1. 自定义数据 ==========

# 左图：堆叠柱状图数据
kv_cache_budgets_labels = [256, 512, 1024, 2048, 4096, 'Full Cache']  # KV cache budget标签
kv_cache_budgets = np.arange(len(kv_cache_budgets_labels))  # 等距位置：0, 1, 2, 3, 4, 5

# 不同操作的延迟（请替换为真实数据）
latency_attn = [0.412, 0.426, 0.432, 0.457, 0.492, 0.883]  # Full Cache的attn延迟
latency_mlp = [0.092, 0.091, 0.093, 0.093, 0.091, 0.095]   # Full Cache的mlp延迟
latency_t2o = [0.052, 0.063, 0.119, 0.188, 0.286, 0.000]   # Full Cache无需T2O操作

# 右图:三个数据集的折线数据(请替换为真实数据)
x_evicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # number of evicted thoughts

# Budget参数(示例数据,请替换为真实数据)
budget_value = 2048 # 可以根据需要修改

aime_values = [50.0, 49.2, 48.5, 47.8, 46.5, 44.0, 38.5, 28.0, 15.5, 6.0]
livecodebench_values = [32.14, 31.5, 30.8, 30.0, 28.5, 26.0, 21.5, 14.0, 8.5, 4.5]
gpqa_values = [49.1, 48.5, 47.8, 47.0, 45.5, 43.0, 38.5, 31.0, 23.5, 17.0]

# 生成置信区间(标准差)
np.random.seed(42)  # 设置随机种子以保证可重复性
aime_std = np.random.uniform(0.8, 2.5, len(x_evicted))
livecodebench_std = np.random.uniform(0.6, 2.0, len(x_evicted))
gpqa_std = np.random.uniform(0.7, 2.3, len(x_evicted))
        
# ========== 2. 创建画布和子图 ==========
fig, (ax1, ax2) = plt.subplots(
    1, 2,          # 2 行 1 列两个子图
    figsize=(18, 8),  # 图像大小
)

# ========== 3. 绘制堆叠柱状图（子图1） ==========
width = 0.6
# 计算堆叠的底部位置
bottom1 = np.array(latency_attn)
bottom2 = bottom1 + np.array(latency_mlp)

ax1.bar(kv_cache_budgets, latency_attn, width, label='Attn', color='#8eaedf')
ax1.bar(kv_cache_budgets, latency_mlp, width, bottom=bottom1, 
        label='MLP', color='#f4b484')
ax1.bar(kv_cache_budgets, latency_t2o, width, 
        bottom=bottom2, label='T2O', color='#a2aadb')
ax1.set_xlabel("KV Cache Budget", fontsize=18)
ax1.set_ylabel("Latency (ms)", fontsize=18)
ax1.set_xticks(kv_cache_budgets)  # 设置x轴刻度位置
ax1.set_xticklabels(kv_cache_budgets_labels)  # 设置x轴刻度标签
ax1.set_title("(a) Overhead across Different KV Cache Budgets during Decoding", fontsize=20, y=-0.15)  # 标题放在底部
ax1.tick_params(axis='both', labelsize=16)
# ax1.set_xlim([-0.5, 4.5])  # 设置x轴范围（可选）
ax1.set_ylim([0, 1.0])  # 设置y轴范围（可选）
ax1.legend(fontsize=16, title=f'Full Length: 16K', title_fontsize=16)
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

# ========== 4. 绘制折线图(子图2) ==========
# 绘制置信区间(阴影)
ax2.fill_between(x_evicted, 
                 np.array(aime_values) - aime_std, 
                 np.array(aime_values) + aime_std, 
                 alpha=0.2, color='#f4b484')
ax2.fill_between(x_evicted, 
                 np.array(livecodebench_values) - livecodebench_std, 
                 np.array(livecodebench_values) + livecodebench_std, 
                 alpha=0.2, color='#8eaedf')
ax2.fill_between(x_evicted, 
                 np.array(gpqa_values) - gpqa_std, 
                 np.array(gpqa_values) + gpqa_std, 
                 alpha=0.2, color='#a2aadb')

# 绘制折线
ax2.plot(x_evicted, aime_values, marker="o", label="AIME24", linewidth=2, color='#f4b484')
ax2.plot(x_evicted, livecodebench_values, marker="s", label="LiveCodeBench", linewidth=2, color='#8eaedf')
ax2.plot(x_evicted, gpqa_values, marker="^", label="GPQA", linewidth=2, color='#a2aadb')
ax2.set_xlabel("Number of Evicted Thoughts", fontsize=18)
ax2.set_ylabel("Accuracy (%)", fontsize=18)
ax2.set_xticks(x_evicted)  # 设置x轴刻度
ax2.set_title("(b) Accuracy v.s. Number of Evicted Thoughts", fontsize=20, y=-0.15)  # 标题放在底部
ax2.tick_params(axis='both', labelsize=16)
# ax2.set_xlim([-0.5, 10.5])  # 设置x轴范围(可选)
# ax2.set_ylim([40, 70])  # 设置y轴范围(可选)
ax2.legend(fontsize=16, title=f'Budget: {budget_value}', title_fontsize=16)
ax2.grid(True, linestyle="--", alpha=0.5)

# ========== 5. 调整布局并保存为 PDF ==========
plt.tight_layout()
plt.savefig("exp_ablation.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("已保存为 exp_ablation.pdf")
