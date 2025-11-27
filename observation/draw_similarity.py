
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import argparse


def draw_heat_map(model, num_layers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    attn_weights = torch.load(f"/home/yangx/zmw/ReasoningPathCompression/observation/sim_heat_map_token/{model}/similarity_layer_24.pt")
    attn_weights_1 = attn_weights[750:850, 750:850]
    attn_weights_2 = attn_weights[1575:1675, 1575:1675]

    sns.heatmap(attn_weights_1.detach().to(torch.float).cpu().numpy(), cmap='Blues', vmin=0, vmax=1, 
            xticklabels=False, yticklabels=False, square=True, ax=ax1)
    ax1.set_title('Cosine Similarity (L24 H3)', fontsize=18)
    ax1.set_xlabel('Key Position', fontsize=16)
    ax1.set_ylabel('Key Position', fontsize=16)

    sns.heatmap(attn_weights_2.detach().to(torch.float).cpu().numpy(), cmap='Blues', vmin=0, vmax=1, 
            xticklabels=False, yticklabels=False, square=True, ax=ax2)
    ax2.set_title('Cosine Similarity (L43 H6)', fontsize=18)
    ax2.set_xlabel('Key Position', fontsize=16)
    ax2.set_ylabel('Key Position', fontsize=16)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    import os
    folder_path = f'/home/yangx/zmw/ReasoningPathCompression/observation/'
    os.makedirs(folder_path, exist_ok=True)

    plt.savefig(f"{folder_path}/method_cossim.pdf")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Draw attention heat maps")
    parser.add_argument("--model", type=str, required=True, help="llama3 qwen2 qwq")
    parser.add_argument("--num_layers", type=int, required=True, help="The total layers of the model")
    args = parser.parse_args()

    draw_heat_map(args.model, args.num_layers)