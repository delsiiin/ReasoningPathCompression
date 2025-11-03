import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import os
import seaborn as sns
import torch

# Data for throughput and memory plots
generation_lengths_llama = [1024, 2048, 4096, 8192, 16384, 32768]
generation_lengths_qwq = [1024, 2048, 4096, 8192, 16384]
memory_usage_llama_8b = [18.83, 21.24, 26.08, 35.74, 55.07, None]  # None represents 'OOM' at 32768
throughput_llama_8b = [385.13, 370.83, 298.90, 208.57, 127.83, 0]  # Throughput data (tokens/s)
memory_usage_qwq_32b = [68.14, 70.42, 74.99, 84.11, None]  # None represents 'OOM' at 32768
throughput_qwq_32b = [88.67, 83.95, 74.05, 60.16, 0]  # Throughput data (tokens/s)


# Create equally spaced x positions
x_positions_llama = range(len(generation_lengths_llama))
x_positions_qwq = range(len(generation_lengths_qwq))

# Create 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# First subplot: LLaMA3 Attention Heatmap
attn_weights_llama = torch.load(f"/home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/llama3/attn_weights_layer_18.pt")
sns.heatmap(attn_weights_llama.detach().to(torch.float).cpu().numpy(), cmap='Blues', vmin=0, vmax=0.01, 
            xticklabels=False, yticklabels=False, square=True, ax=ax1)
ax1.set_title('R1-Llama-8B Attention Heatmap (Layer 18)', fontsize=14)
ax1.set_xlabel('Key Position')
ax1.set_ylabel('Query Position')

# Second subplot: Qwen3 Attention Heatmap
attn_weights_qwen = torch.load(f"/home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/qwq/attn_weights_layer_42.pt")
sns.heatmap(attn_weights_qwen.detach().to(torch.float).cpu().numpy(), cmap='Blues', vmin=0, vmax=0.01, 
            xticklabels=False, yticklabels=False, square=True, ax=ax2)
ax2.set_title('QwQ-32B Attention Heatmap (Layer 42)', fontsize=14)
ax2.set_xlabel('Key Position')
ax2.set_ylabel('Query Position')

# Third subplot: Memory Usage and Throughput
bars3 = ax3.bar(x_positions_llama[:-1], memory_usage_llama_8b[:-1], color=(245/255, 180/255, 130/255), alpha=0.7)
ax3.bar(x_positions_llama[-1], 100, color='gray', hatch='//', alpha=0.7)  # for the OOM bar
ax3.text(x_positions_llama[-1], 100, "OOM", ha='center', va='center', fontsize=12, color='black')

# Labeling for third subplot
ax3.set_title('R1-Llama-8B (Batch Size 16)', fontsize=14)
ax3.set_xlabel('Generation Length')
ax3.set_ylabel('Peak Memory (GB)')
ax3.set_xticks(x_positions_llama)
ax3.set_xticklabels(generation_lengths_llama)
ax3.tick_params(axis='y')

# Annotate each bar with its value for third subplot
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, height + height * 0.02, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)

# Add throughput line to third subplot
ax3_twin = ax3.twinx()
line3 = ax3_twin.plot(x_positions_llama, throughput_llama_8b, color=(88/255, 142/255, 50/255), marker='o', linewidth=3, markersize=8, label='Throughput')
ax3_twin.set_ylabel('Throughput (Tokens/s)')
ax3_twin.tick_params(axis='y')

# Fourth subplot: Combined Memory and Throughput (alternative view)
bars4 = ax4.bar(x_positions_qwq[:-1], memory_usage_qwq_32b[:-1], color=(245/255, 180/255, 130/255), alpha=0.7, label='Memory Usage')
ax4.bar(x_positions_qwq[-1], 100, color='gray', hatch='//', alpha=0.7)  # for the OOM bar
ax4.text(x_positions_qwq[-1], 100, "OOM", ha='center', va='center', fontsize=12, color='black')

# Labeling for fourth subplot
ax4.set_title('QwQ-32B (Batch Size 8)', fontsize=14)
ax4.set_xlabel('Generation Length')
ax4.set_ylabel('Peak Memory (GB)')
ax4.set_xticks(x_positions_qwq)
ax4.set_xticklabels(generation_lengths_qwq)
ax4.tick_params(axis='y')

# Annotate each bar with its value for fourth subplot
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2, height + height * 0.02, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)

# Add throughput line to fourth subplot
ax4_twin = ax4.twinx()
line4 = ax4_twin.plot(x_positions_qwq, throughput_qwq_32b, color=(88/255, 142/255, 50/255), marker='o', linewidth=3, markersize=8, label='Throughput')
ax4_twin.set_ylabel('Throughput (Tokens/s)')
ax4_twin.tick_params(axis='y')

plt.tight_layout()
plt.show()
plt.savefig("intro_throughput_mem.pdf", dpi=300, bbox_inches='tight')
