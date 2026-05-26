import argparse
import functools
import http.server
import json
import os
import socketserver
from datetime import datetime

import numpy as np

from plot_token_entropy import (
    build_step_stats,
    find_subsequence,
    load_entropy_values,
    load_first_jsonl_record,
)


MODEL_DEFAULTS = {
    "llama": ("llama3", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    "llama3": ("llama3", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    "qwen": ("qwen2", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    "qwen2": ("qwen2", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    "qwen3": ("qwen3", "Qwen/Qwen3-30B-A3B"),
    "gpt": ("oss", "openai/gpt-oss-20b"),
    "oss": ("oss", "openai/gpt-oss-20b"),
    "gpt_oss": ("oss", "openai/gpt-oss-20b"),
}


def resolve_defaults(args):
    folder_name, tokenizer_name = MODEL_DEFAULTS.get(args.model_type, MODEL_DEFAULTS["llama3"])
    if args.compression:
        folder_name = f"{folder_name}_{args.compression}"
        if args.input_file is None:
            args.input_file = f"observation/output_{args.compression}.jsonl"

    if args.tensor_path is None:
        args.tensor_path = f"observation/token_entropy/{folder_name}/entropy.pt"
    if args.output is None:
        args.output = f"observation/token_entropy/{folder_name}/step_entropy_app.html"
    if args.input_file is None:
        args.input_file = "observation/output.jsonl"
    if args.tokenizer_name is None:
        args.tokenizer_name = tokenizer_name
    return args


def load_tokenizer(tokenizer_name):
    if not tokenizer_name:
        return None, "未提供 tokenizer_name，step 文本使用 decoded_output 近似切分。"

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True), None
    except Exception as exc:
        return None, f"无法加载 tokenizer ({tokenizer_name})，step 文本使用 decoded_output 近似切分: {exc}"


def load_token_metadata(record, tokenizer, notes):
    generated_token_ids = record.get("generated_token_ids")
    newline_token_ids = record.get("newline_token_ids")

    if generated_token_ids is not None and newline_token_ids is not None:
        return [int(x) for x in generated_token_ids], [int(x) for x in newline_token_ids]

    if tokenizer is not None and record.get("decoded_output"):
        notes.append("input_file 缺少 token 元数据，已用 tokenizer 对 decoded_output 重新 tokenize。")
        token_ids = tokenizer(record["decoded_output"], add_special_tokens=False)["input_ids"]
        newline_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]
        return token_ids, newline_ids

    raise ValueError("input_file 中缺少 generated_token_ids/newline_token_ids，且无法回退 tokenize。")


def maybe_truncate_think(token_ids, values, tokenizer, skip_answer, notes):
    if not skip_answer:
        return token_ids, values
    if tokenizer is None:
        notes.append("--skip_answer 已开启，但 tokenizer 不可用，未截断 answer 部分。")
        return token_ids, values

    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    start = find_subsequence(token_ids, end_think_ids)
    if start < 0:
        notes.append("未找到 </think> token，保留全部 entropy。")
        return token_ids, values

    end = start + len(end_think_ids)
    notes.append(f"已按 </think> 截断到 token 位置 {end}。")
    return token_ids[:end], values[:end]


def split_decoded_output(text):
    segments = []
    start = 0
    idx = 0
    while idx < len(text):
        if text[idx] == "\n":
            idx += 1
            while idx < len(text) and text[idx] == "\n":
                idx += 1
            segments.append(text[start:idx])
            start = idx
        else:
            idx += 1

    if start < len(text):
        segments.append(text[start:])
    return segments


def attach_step_text(stats, token_ids, tokenizer, decoded_output, notes):
    if tokenizer is not None:
        for item in stats:
            item["text"] = tokenizer.decode(
                token_ids[item["start"]:item["end"]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        return

    segments = split_decoded_output(decoded_output or "")
    if len(segments) != len(stats):
        notes.append(f"decoded_output 近似切分得到 {len(segments)} 段，与 {len(stats)} 个 step 不完全一致。")

    for idx, item in enumerate(stats):
        item["text"] = segments[idx] if idx < len(segments) else ""


def make_payload(stats, values, token_ids, record, args, notes):
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    for item in stats:
        item["token_count"] = item["end"] - item["start"]

    summary = {
        "model_family": record.get("model_family"),
        "input_file": args.input_file,
        "tensor_path": args.tensor_path,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "token_count": min(len(values), len(token_ids)),
        "step_count": len(stats),
        "output_length": record.get("output_length"),
        "context_length": record.get("context_length"),
        "entropy_min": float(finite_values.min()) if len(finite_values) else None,
        "entropy_max": float(finite_values.max()) if len(finite_values) else None,
        "entropy_mean": float(finite_values.mean()) if len(finite_values) else None,
        "entropy_std": float(finite_values.std()) if len(finite_values) else None,
    }
    return {"summary": summary, "notes": notes, "steps": stats}


def json_for_html(payload):
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def build_html(payload):
    data_json = json_for_html(payload)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step Entropy Explorer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef2f4;
      --text: #20242a;
      --muted: #67717d;
      --line: #d9dee5;
      --accent: #167a72;
      --accent-2: #b6406b;
      --accent-3: #5b6f21;
      --accent-4: #6a5acd;
      --focus: #0f766e;
      --warn: #8a5a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 22px 28px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      max-width: 1400px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--surface-2);
      min-width: 0;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .stat strong {{
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 17px;
      font-weight: 650;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(520px, 1.2fr) minmax(360px, 0.8fr);
      gap: 16px;
      padding: 16px 28px 28px;
      max-width: 1500px;
    }}
    section {{
      min-width: 0;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }}
    .panel + .panel {{ margin-top: 16px; }}
    .panel-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    h2 {{
      margin: 0;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 14px;
    }}
    .chart {{
      min-height: 190px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #ffffff;
    }}
    .chart-title {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
      font-weight: 650;
    }}
    .chart-title small {{ color: var(--muted); font-weight: 500; }}
    svg {{ display: block; width: 100%; height: 145px; }}
    .axis {{ stroke: #c8d0d8; stroke-width: 1; }}
    .line {{ fill: none; stroke-width: 2.2; }}
    .point {{ cursor: pointer; stroke: #ffffff; stroke-width: 1.5; }}
    .point.selected {{ stroke: #20242a; stroke-width: 2.4; }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }}
    input[type="search"] {{
      width: 100%;
      min-width: 160px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--text);
      background: #ffffff;
      font: inherit;
    }}
    .table-wrap {{ max-height: 460px; overflow: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #fbfcfd;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr {{ cursor: pointer; }}
    tbody tr:hover {{ background: #f1f7f6; }}
    tbody tr.selected {{ background: #dff0ed; }}
    .empty {{
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }}
    .text-panel {{
      min-height: 520px;
      display: flex;
      flex-direction: column;
    }}
    .step-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .meta-item {{
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .meta-item strong {{
      display: block;
      color: var(--text);
      font-size: 15px;
    }}
    .step-text {{
      flex: 1;
      min-height: 320px;
      margin: 0;
      padding: 14px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #ffffff;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      padding: 12px 14px;
      border-top: 1px solid var(--line);
      background: #fbfcfd;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 11px;
      color: var(--text);
      background: #ffffff;
      font: inherit;
      cursor: pointer;
    }}
    button:hover, button:focus {{ border-color: var(--focus); outline: none; }}
    .notes {{
      margin: 0;
      padding: 10px 14px;
      color: var(--warn);
      border-bottom: 1px solid var(--line);
      background: #fff8e8;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; padding: 14px; }}
      header {{ padding: 18px 14px 10px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 620px) {{
      .charts {{ grid-template-columns: 1fr; }}
      .summary {{ grid-template-columns: 1fr; }}
      .step-meta {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Step Entropy Explorer</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <section>
      <div class="panel">
        <div class="panel-head">
          <h2>Metrics</h2>
          <span id="filter-count"></span>
        </div>
        <div id="notes"></div>
        <div class="charts" id="charts"></div>
      </div>
      <div class="panel">
        <div class="toolbar">
          <input id="search" type="search" placeholder="Search step text or id">
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th data-sort="step_id">Step</th>
                <th data-sort="start">Range</th>
                <th data-sort="token_count">Tokens</th>
                <th data-sort="mean">Mean</th>
                <th data-sort="max">Max</th>
                <th data-sort="min">Min</th>
                <th data-sort="std">Std</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
    </section>
    <section>
      <div class="panel text-panel">
        <div class="panel-head">
          <h2 id="step-title">Step</h2>
        </div>
        <div class="step-meta" id="step-meta"></div>
        <pre class="step-text" id="step-text"></pre>
        <div class="actions">
          <button id="prev">Previous</button>
          <button id="next">Next</button>
        </div>
      </div>
    </section>
  </main>
  <script type="application/json" id="app-data">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('app-data').textContent);
    const metrics = [
      ['mean', 'Mean Entropy', '#167a72'],
      ['max', 'Max Entropy', '#b6406b'],
      ['min', 'Min Entropy', '#5b6f21'],
      ['std', 'Std Entropy', '#6a5acd'],
    ];
    const state = {{ selectedId: payload.steps[0]?.step_id ?? null, sortKey: 'step_id', sortDir: 1, query: '' }};

    const fmt = (value) => Number.isFinite(value) ? value.toFixed(6) : 'n/a';
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));

    function filteredSteps() {{
      const q = state.query.trim().toLowerCase();
      let rows = payload.steps.filter(step => !q || String(step.step_id).includes(q) || (step.text || '').toLowerCase().includes(q));
      rows = rows.slice().sort((a, b) => {{
        const av = a[state.sortKey];
        const bv = b[state.sortKey];
        if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * state.sortDir;
        return String(av).localeCompare(String(bv)) * state.sortDir;
      }});
      return rows;
    }}

    function selectedStep() {{
      return payload.steps.find(step => step.step_id === state.selectedId) || payload.steps[0] || null;
    }}

    function renderSummary() {{
      const s = payload.summary;
      const items = [
        ['Model', s.model_family ?? 'unknown'],
        ['Steps', s.step_count],
        ['Tokens', s.token_count],
        ['Entropy Range', `${{fmt(s.entropy_min)}} - ${{fmt(s.entropy_max)}}`],
        ['Generated', s.generated_at],
        ['Input', s.input_file],
        ['Tensor', s.tensor_path],
        ['Context Length', s.context_length ?? 'n/a'],
        ['Output Length', s.output_length ?? 'n/a'],
      ];
      document.getElementById('summary').innerHTML = items.map(([label, value]) =>
        `<div class="stat"><span>${{esc(label)}}</span><strong title="${{esc(value)}}">${{esc(value)}}</strong></div>`
      ).join('');
      document.getElementById('notes').innerHTML = payload.notes.length
        ? `<p class="notes">${{payload.notes.map(esc).join('<br>')}}</p>`
        : '';
    }}

    function renderCharts(rows) {{
      const charts = document.getElementById('charts');
      if (!rows.length) {{
        charts.innerHTML = '<div class="empty">No matching steps</div>';
        return;
      }}
      charts.innerHTML = metrics.map(([key, label, color]) => chartSvg(rows, key, label, color)).join('');
      charts.querySelectorAll('[data-step]').forEach(el => {{
        el.addEventListener('click', () => selectStep(Number(el.dataset.step)));
      }});
    }}

    function chartSvg(rows, key, label, color) {{
      const width = 520, height = 145, pad = 28;
      const xs = rows.map(row => row.step_id);
      const ys = rows.map(row => row[key]);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      if (minY === maxY) {{ minY -= 1; maxY += 1; }}
      const xScale = x => pad + ((x - minX) / Math.max(1, maxX - minX)) * (width - pad * 2);
      const yScale = y => height - pad - ((y - minY) / Math.max(1e-9, maxY - minY)) * (height - pad * 2);
      const points = rows.map(row => [xScale(row.step_id), yScale(row[key]), row]);
      const path = points.map((p, idx) => `${{idx ? 'L' : 'M'}}${{p[0].toFixed(2)}} ${{p[1].toFixed(2)}}`).join(' ');
      const circles = points.map(([x, y, row]) =>
        `<circle class="point${{row.step_id === state.selectedId ? ' selected' : ''}}" data-step="${{row.step_id}}" cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="4.2" fill="${{color}}"><title>step ${{row.step_id}}: ${{fmt(row[key])}}</title></circle>`
      ).join('');
      return `<div class="chart">
        <div class="chart-title"><span>${{esc(label)}}</span><small>${{fmt(minY)}} - ${{fmt(maxY)}}</small></div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{esc(label)}}">
          <line class="axis" x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}"></line>
          <line class="axis" x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height-pad}}"></line>
          <path class="line" d="${{path}}" stroke="${{color}}"></path>
          ${{circles}}
        </svg>
      </div>`;
    }}

    function renderTable(rows) {{
      document.getElementById('filter-count').textContent = `${{rows.length}} / ${{payload.steps.length}} steps`;
      document.getElementById('rows').innerHTML = rows.map(step => `
        <tr data-step="${{step.step_id}}" class="${{step.step_id === state.selectedId ? 'selected' : ''}}">
          <td>Step ${{step.step_id}}</td>
          <td>${{step.start}}:${{step.end}}</td>
          <td>${{step.token_count}}</td>
          <td>${{fmt(step.mean)}}</td>
          <td>${{fmt(step.max)}}</td>
          <td>${{fmt(step.min)}}</td>
          <td>${{fmt(step.std)}}</td>
        </tr>
      `).join('');
      document.querySelectorAll('#rows tr').forEach(row => {{
        row.addEventListener('click', () => selectStep(Number(row.dataset.step)));
      }});
    }}

    function renderSelected() {{
      const step = selectedStep();
      if (!step) return;
      document.getElementById('step-title').textContent = `Step ${{step.step_id}}`;
      document.getElementById('step-meta').innerHTML = [
        ['Range', `${{step.start}}:${{step.end}}`],
        ['Tokens', step.token_count],
        ['Mean / Std', `${{fmt(step.mean)}} / ${{fmt(step.std)}}`],
        ['Min / Max', `${{fmt(step.min)}} / ${{fmt(step.max)}}`],
      ].map(([label, value]) => `<div class="meta-item">${{esc(label)}}<strong>${{esc(value)}}</strong></div>`).join('');
      document.getElementById('step-text').textContent = step.text || '';
    }}

    function render() {{
      const rows = filteredSteps();
      renderCharts(rows);
      renderTable(rows);
      renderSelected();
    }}

    function selectStep(stepId) {{
      state.selectedId = stepId;
      render();
    }}

    function move(delta) {{
      const rows = filteredSteps();
      if (!rows.length) return;
      const idx = Math.max(0, rows.findIndex(step => step.step_id === state.selectedId));
      const next = rows[Math.min(rows.length - 1, Math.max(0, idx + delta))];
      selectStep(next.step_id);
    }}

    document.getElementById('search').addEventListener('input', event => {{
      state.query = event.target.value;
      const rows = filteredSteps();
      if (rows.length && !rows.some(step => step.step_id === state.selectedId)) {{
        state.selectedId = rows[0].step_id;
      }}
      render();
    }});
    document.querySelectorAll('th[data-sort]').forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.sort;
        state.sortDir = state.sortKey === key ? -state.sortDir : 1;
        state.sortKey = key;
        render();
      }});
    }});
    document.getElementById('prev').addEventListener('click', () => move(-1));
    document.getElementById('next').addEventListener('click', () => move(1));

    renderSummary();
    render();
  </script>
</body>
</html>
"""


def build_app(args):
    notes = []
    values = load_entropy_values(args.tensor_path, args.dict_key)
    record = load_first_jsonl_record(args.input_file)
    tokenizer, tokenizer_note = load_tokenizer(args.tokenizer_name)
    if tokenizer_note:
        notes.append(tokenizer_note)

    token_ids, newline_token_ids = load_token_metadata(record, tokenizer, notes)
    token_ids, values = maybe_truncate_think(token_ids, values, tokenizer, args.skip_answer, notes)
    stats = build_step_stats(values, token_ids, newline_token_ids)
    attach_step_text(stats, token_ids, tokenizer, record.get("decoded_output", ""), notes)

    payload = make_payload(stats, values, token_ids, record, args, notes)
    html_text = build_html(payload)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"网页已保存到: {args.output}")
    print(f"Step数: {len(stats)}")
    return args.output


def serve_app(output_path, host, port):
    directory = os.path.abspath(os.path.dirname(output_path) or ".")
    filename = os.path.basename(output_path)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/{filename}"
        print(f"服务已启动: {url}")
        print("按 Ctrl+C 停止服务")
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="生成 step entropy 静态网页应用")
    parser.add_argument("--model_type", default="llama3", help="模型类型: llama3/qwen2/qwen3/oss")
    parser.add_argument("--compression", help="压缩方法后缀，例如 h2o/snapkv/streamingllm")
    parser.add_argument(
        "--tensor_path",
        default=None,
        help="输入的 entropy 张量路径",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        default=None,
        help="包含 decoded_output/generated_token_ids/newline_token_ids 的 JSONL 文件",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出 HTML 路径",
    )
    parser.add_argument("--tokenizer_name", "-t", default=None)
    parser.add_argument("--dict-key", help="如果 tensor_path 保存的是字典，指定取值键名")
    parser.add_argument("--skip_answer", action="store_true", help="遇到 </think> 后截断 answer 部分")
    parser.add_argument("--serve", action="store_true", help="生成后启动本地 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 服务监听地址")
    parser.add_argument("--port", type=int, default=8765, help="HTTP 服务端口")
    args = resolve_defaults(parser.parse_args())
    output_path = build_app(args)
    if args.serve:
        serve_app(output_path, args.host, args.port)


if __name__ == "__main__":
    main()
