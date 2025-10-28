
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import argparse


def draw_heat_map(model, num_layers):
    for layer_idx in range(num_layers):
        attn_weights = torch.load(f"/home/yangx/ReasoningPathCompression/observation/attn_heat_map_token/{model}/attn_weights_layer_{layer_idx}.pt")
        print(attn_weights.shape)
        plt.figure(figsize=(12, 10))
        sns.heatmap(attn_weights.detach().to(torch.float).cpu().numpy(), cmap='Reds', vmin=0, vmax=0.01, xticklabels=True, yticklabels=True, square=True)
        plt.title(f'Attention Heatmap (Prefill + Decode)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

        import os
        folder_path = f'/home/yangx/ReasoningPathCompression/observation/attn_heat_map_token/token_wise/{model}'
        os.makedirs(folder_path, exist_ok=True)

        plt.savefig(f"{folder_path}/{layer_idx}.pdf")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Draw attention heat maps")
    parser.add_argument("--model", type=str, required=True, help="llama3 qwen2 qwq")
    parser.add_argument("--num_layers", type=int, required=True, help="The total layers of the model")
    args = parser.parse_args()

    draw_heat_map(args.model, args.num_layers)