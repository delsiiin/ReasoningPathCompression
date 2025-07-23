import math
import torch


#################################################################
###################### kv cache utilities #######################
#################################################################
def compute_attention_scores(query_states, key_states, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    else:
        # shape: [batch_size, kv_heads, query_group_size, q_len, head_dim]
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )

        # shape: [batch_size, kv_heads, 1, kv_len, head_dim]
        key_states = key_states.unsqueeze(2)

        # shape: [batch_size, kv_heads, query_group_size, q_len, kv_len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 4)
        ) / math.sqrt(head_dim)

        # apply pooling over query_group_size dimension
        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
        else:
            raise ValueError("Pooling method not supported")

    return attn_weights


def cal_similarity(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    k = key_states[0]
    num_heads = k.shape[0]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    # shape: [num_heads, seq_len, seq_len]
    similarity_mask = similarity_cos > threshold

    seq_len = similarity_mask.size(-1)
    k = int(seq_len * retain_ratio)

    indices = torch.where(
        similarity_mask,
        torch.arange(similarity_mask.size(-1), device=similarity_mask.device),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    # find the last True index in each row
    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]

    # find the first True index in each row
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]

    # keep the last_percent% elements
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]

    # keep the first_percent% elements
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]

    # create indices for zeroing
    batch_idx = (
        torch.arange(num_heads).unsqueeze(1).repeat(1, similarity_retain.size(1))
    )
    seq_idx = torch.arange(similarity_retain.size(1)).unsqueeze(0).repeat(num_heads, 1)

    # zero the specified positions in similarity_cos
    similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

    return similarity_cos.mean(dim=1).softmax(dim=-1)


#################################################################
################### visualization utilities #####################
#################################################################

# Visualize the token eviction pattern for a given head
def visualize_token_eviction(
    output_token_ids, kept_token_indices, tokenizer, head_idx=0
):
    """
    Visualize which tokens are kept vs evicted for a given head

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices: shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
    """
    from IPython.display import HTML

    # Get the kept indices for the specified head
    kept_indices = set(kept_token_indices[head_idx].tolist())

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Build HTML with different colors for kept vs evicted tokens
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")  # Remove space marker
            .replace("Ċ", "\n")  # Convert newline marker to actual newline
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        if idx in kept_indices:
            # Kept tokens in green with bold
            html_parts.append(
                f'<span style="color: green; font-weight: bold;">{token}</span>'
            )
        else:
            # Evicted tokens in gray and lighter
            html_parts.append(f'<span style="color: #999999;">{token}</span>')

    # Join without spaces (since we're now handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)

# Visualize the token eviction pattern for a given heads at each compression step
def visualize_multistep_token_eviction(
    output_token_ids, kept_token_indices_list, tokenizer, head_idx=0, step_idx=-1
):
    """
    Visualize which tokens are kept at each compression step with different colors.
    Later steps are shown in more vibrant colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
    """
    from IPython.display import HTML

    # Get the kept indices for each step for the specified head
    kept_indices_by_step = [
        set(indices[head_idx].tolist()) for indices in kept_token_indices_list
    ]
    num_steps = len(kept_indices_by_step) if step_idx == -1 else 1

    # Generate colors using a distinct color spectrum
    def get_color(step):
        # Use a color spectrum for better distinction between steps
        if num_steps <= 1:
            return "#3498db"  # Default blue if only one step

        # Define a set of distinct colors
        colors = [
            "#e74c3c",  # Red
            "#3498db",  # Blue
            "#2ecc71",  # Green
            "#f39c12",  # Orange
            "#9b59b6",  # Purple
            "#1abc9c",  # Teal
            "#d35400",  # Dark Orange
            "#2980b9",  # Dark Blue
            "#27ae60",  # Dark Green
            "#8e44ad",  # Dark Purple
        ]

        if num_steps <= len(colors):
            # If we have fewer steps than colors, use the colors directly
            return colors[step % len(colors)]
        else:
            # For more steps than colors, interpolate between colors
            # Map step to a position in the color spectrum
            position = (step / (num_steps - 1)) * (len(colors) - 1)
            idx1 = int(position)
            idx2 = min(idx1 + 1, len(colors) - 1)
            fraction = position - idx1

            # Get the two colors to interpolate between
            color1 = colors[idx1]
            color2 = colors[idx2]

            # Convert hex to RGB
            r1, g1, b1 = (
                int(color1[1:3], 16),
                int(color1[3:5], 16),
                int(color1[5:7], 16),
            )
            r2, g2, b2 = (
                int(color2[1:3], 16),
                int(color2[3:5], 16),
                int(color2[5:7], 16),
            )

            # Interpolate
            r = int(r1 * (1 - fraction) + r2 * fraction)
            g = int(g1 * (1 - fraction) + g2 * fraction)
            b = int(b1 * (1 - fraction) + b2 * fraction)

            return f"#{r:02x}{g:02x}{b:02x}"

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        latest_step = -1
        if step_idx == -1:
            # Find the latest step (if any) where this token was kept
            for step, kept_indices in enumerate(kept_indices_by_step[::-1]):
                if idx in kept_indices:
                    latest_step = num_steps - step
                    break

        elif idx in kept_indices_by_step[step_idx]:
            latest_step = num_steps

        # Color the token based on its latest appearance
        if latest_step >= 0:
            color = get_color(latest_step)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)


# Visualize the token eviction pattern for all heads at each compression step
def visualize_multistep_token_eviction_by_head(
    output_token_ids, kept_token_indices_list, tokenizer, step_idx, aggregate=False
):
    """
    Visualize which tokens are kept by which heads with different colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
        aggregate: when set to False, later heads will cover previous heads. when set to `True`, will compute how many times a token are covered by a head.
    """
    from IPython.display import HTML

    # Generate colors using a distinct color spectrum
    def get_color(idx, aggregate):
        # Define a set of distinct colors
        if not aggregate:
            colors = [
                "#3498db",  # Blue
                "#f39c12",  # Orange
                "#9b59b6",  # Purple
                "#1abc9c",  # Teal
                "#d35400",  # Dark Orange
                "#2980b9",  # Dark Blue
                "#27ae60",  # Dark Green
                "#8e44ad",  # Dark Purple
            ]
        else:
            # colors = [
            #     "#D6EAF8",  # Very Light Blue
            #     "#AED6F1",  # Light Blue
            #     "#85C1E9",  # Medium Light Blue
            #     "#5DADE2",  # Medium Blue
            #     "#3498DB",  # Blue
            #     "#2E86C1",  # Medium Dark Blue
            #     "#2874A6",  # Dark Blue
            #     "#1B4F72",  # Very Dark Blue
            # ]
            colors = [
                "#E65100",  # 深橙色(起点)
                "#D84315",  # 橙红色
                "#C62828",  # 红褐色
                "#B71C1C",  # 深红褐色
                "#A52A00",  # 深橙褐色
                "#8B2500",  # 赭石色
                "#7C2000",  # 深褐色
                "#6B1D00"   # 最深的橙褐色
            ]
        return colors[idx]

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Get kept token id list
    token_indices_lst = kept_token_indices_list[
        step_idx
    ]  # shape: (kv_head, num_kept_tokens)
    token_indices_dict = {
        i: set(token_indices_lst[i].tolist()) for i in range(token_indices_lst.shape[0])
    }

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        color_idx = -1
        for head_idx, kept_token_set in token_indices_dict.items():
            if idx in kept_token_set:
                if aggregate:
                    color_idx += 1
                else:
                    color_idx = head_idx

        # Color the token based on its latest appearance
        if color_idx >= 0:
            color = get_color(color_idx, aggregate)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)


# Visualize the token eviction score for all heads at each compression step
def visualize_multistep_token_eviction_score_by_head(
    output_token_ids, kept_token_indices_list, score_list, tokenizer, step_idx, head_idx
):
    """
    Visualize which tokens are kept by which heads with different colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
        aggregate: when set to False, later heads will cover previous heads. when set to `True`, will compute how many times a token are covered by a head.
    """
    from IPython.display import HTML

    # Generate colors using the common blue to yellow heatmap color spectrum
    def get_color(score):
        # Define the blue and yellow colors
        colors = [
            "#D6EAF8",  # Very Light Blue
            "#AED6F1",  # Light Blue
            "#85C1E9",  # Medium Light Blue
            "#5DADE2",  # Medium Blue
            "#3498DB",  # Blue
            "#2E86C1",  # Medium Dark Blue
            "#2874A6",  # Dark Blue
            "#1B4F72",  # Very Dark Blue
        ]

        if score <= 0:
            return colors[0]

        # Calculate the position of the step within the range of colors
        position = score * (len(colors) - 1)

        # Determine the indices for interpolation
        idx1 = int(position)
        idx2 = min(idx1 + 1, len(colors) - 1)
        fraction = position - idx1

        # Get the two colors to interpolate between
        color1 = colors[idx1]
        color2 = colors[idx2]

    # Convert hex to RGB for color1
        r1, g1, b1 = (
            int(color1[1:3], 16),
            int(color1[3:5], 16),
            int(color1[5:7], 16),
        )
        # Convert hex to RGB for color2
        r2, g2, b2 = (
            int(color2[1:3], 16),
            int(color2[3:5], 16),
            int(color2[5:7], 16),
        )

        # Interpolate between the two colors
        r = int(r1 * (1 - fraction) + r2 * fraction)
        g = int(g1 * (1 - fraction) + g2 * fraction)
        b = int(b1 * (1 - fraction) + b2 * fraction)

        # Return the interpolated color as a hex code
        return f"#{r:02x}{g:02x}{b:02x}"


    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Get kept token id list
    token_indices_lst = kept_token_indices_list[
        step_idx
    ]  # shape: (kv_head, num_kept_tokens)
    token_indices_dict = {
        i: token_indices_lst[i].tolist() for i in range(token_indices_lst.shape[0])
    }

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        score = -1
        # for head_idx, kept_token_set in token_indices_dict.items():
            # if idx in kept_token_set:

        # 定位idx在kept_token_set的index
        if idx in token_indices_dict[head_idx]:
            index = token_indices_dict[head_idx].index(idx)
        else:
            # 处理 idx 不在列表中的情况
            index = -1  # 或者其他合适的值

        if index != -1:
            score = score_list[step_idx][head_idx][index].item()

            # if aggregate:
            #     color_idx += 1
            # else:
            #     color_idx = head_idx

        # Color the token based on its latest appearance
        if score >= 0:
            color = get_color(score)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)