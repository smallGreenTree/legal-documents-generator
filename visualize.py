#!/usr/bin/env python3
"""
Visualise generated artifacts.

Usage:
    poetry run python visualize.py                   # most recent document coverage
    poetry run python visualize.py --all             # all document coverage reports
    poetry run python visualize.py --doc DOC_ID      # specific document coverage
    poetry run python visualize.py --graph           # most recent schema graph
    poetry run python visualize.py --graph DOC_ID    # specific schema graph
"""

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path

OUTPUT_DIR = Path("output")
SCHEMA_DIR = Path("schemas")

LABEL_STYLES = {
    "PERSON":           "background:#fde68a;border-radius:3px;padding:1px 3px;",
    "ORG":              "background:#99f6e4;border-radius:3px;padding:1px 3px;",
    "LOCATION":         "background:#bfdbfe;border-radius:3px;padding:1px 3px;",
    "AMOUNT":           "background:#d9f99d;border-radius:3px;padding:1px 3px;",
    "DATE":             "background:#fecaca;border-radius:3px;padding:1px 3px;",
    "VAT":              "background:#e9d5ff;border-radius:3px;padding:1px 3px;",
    "CASE_REF":         "background:#fed7aa;border-radius:3px;padding:1px 3px;",
    "INITIALS":         "background:#fde68a;border-radius:3px;padding:1px 3px;",
    "TITLE":            "background:#fde68a;border-radius:3px;padding:1px 3px;",
    "NEGATIVE_CONTROL": (
        "border-bottom:2px dashed #ef4444;"
        "padding:1px 3px;"
        "text-decoration:none;"
    ),
}

LEGEND_COLORS = {
    "PERSON":           "#fde68a",
    "ORG":              "#99f6e4",
    "LOCATION":         "#bfdbfe",
    "AMOUNT":           "#d9f99d",
    "DATE":             "#fecaca",
    "VAT":              "#e9d5ff",
    "CASE_REF":         "#fed7aa",
    "NEGATIVE_CONTROL": "#ffffff",
}

NODE_COLORS = {
    "defendant": "#fbbf24",
    "collateral": "#fde68a",
    "charged": "#2dd4bf",
    "associated": "#99f6e4",
}

EDGE_COLORS = {
    "controlled": "#2563eb",
    "directed": "#2563eb",
    "used_as_vehicle": "#2563eb",
    "laundered_through": "#2563eb",
    "instructed": "#dc2626",
    "conspired_with": "#7c3aed",
    "bribed": "#c2410c",
    "invoiced": "#0f766e",
    "subcontracted_to": "#0f766e",
    "received_funds_from": "#059669",
}


def load_doc(doc_dir: Path) -> tuple[str, list[tuple]]:
    """Return (text, gt_rows) from a doc folder."""
    doc_id  = doc_dir.name
    txt     = (doc_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
    gt_path = doc_dir / "groundtruth.tsv"
    rows = []
    if gt_path.exists():
        with open(gt_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append((
                    row["entity_text"],
                    row["label"],
                    row["should_propose"],
                ))
    return txt, rows


def load_schema(schema_path: Path) -> dict:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def annotate(text: str, entities: list[tuple]) -> str:
    """
    Replace entity occurrences in text with <span> tags.
    Handles overlapping spans by processing longest matches first.
    """
    # Sort by length desc so longer matches win
    ents = sorted(entities, key=lambda e: len(e[0]), reverse=True)

    # Build list of (start, end, label, should_propose) non-overlapping matches
    matches: list[tuple[int, int, str, str]] = []
    occupied = set()

    for entity_text, label, should_propose in ents:
        pattern = re.escape(entity_text)
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            span = range(s, e)
            if any(i in occupied for i in span):
                continue
            matches.append((s, e, label, should_propose))
            occupied.update(span)

    if not matches:
        return html.escape(text).replace("\n", "<br>")

    matches.sort(key=lambda x: x[0])

    parts = []
    prev = 0
    for s, e, label, should_propose in matches:
        parts.append(html.escape(text[prev:s]))
        style = LABEL_STYLES.get(label, "")
        tag = "del" if label == "NEGATIVE_CONTROL" else "mark"
        title = f"{label}" + ("" if should_propose == "yes" else " [negative control]")
        parts.append(
            f'<{tag} style="{style}" title="{title}">'
            f'{html.escape(text[s:e])}'
            f'</{tag}>'
        )
        prev = e
    parts.append(html.escape(text[prev:]))

    return "".join(parts).replace("\n", "<br>\n")


def coverage_stats(text: str, entities: list[tuple]) -> list[dict]:
    stats = []
    for entity_text, label, should_propose in entities:
        count = len(re.findall(re.escape(entity_text), text, flags=re.IGNORECASE))
        stats.append({
            "entity":  entity_text,
            "label":   label,
            "propose": should_propose,
            "hits":    count,
        })
    return stats


def legend_html() -> str:
    items = "".join(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{color};border:1px solid #ccc;margin-right:4px;'
        f'vertical-align:middle;border-radius:2px;"></span>{label} &nbsp;'
        for label, color in LEGEND_COLORS.items()
    )
    return f'<div style="margin-bottom:12px;font-size:13px;">{items}</div>'


def render_html(doc_id: str, text: str, entities: list[tuple]) -> str:
    annotated = annotate(text, entities)
    stats     = coverage_stats(text, entities)

    rows_html = "".join(
        f'<tr style="background:{"#f9fafb" if i%2==0 else "white"}">'
        f'<td style="padding:4px 8px;">{html.escape(s["entity"][:80])}</td>'
        f'<td style="padding:4px 8px;">{s["label"]}</td>'
        f'<td style="padding:4px 8px;text-align:center;">{s["propose"]}</td>'
        f'<td style="padding:4px 8px;text-align:center;'
        f'color:{"#16a34a" if s["hits"]>0 else "#dc2626"}">'
        f'{"✓" if s["hits"]>0 else "✗"} {s["hits"]}</td>'
        f'</tr>'
        for i, s in enumerate(stats)
    )

    missing = sum(1 for s in stats if s["hits"] == 0 and s["propose"] == "yes")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{doc_id}</title>
<style>
  body  {{ font-family: monospace; font-size: 14px; margin: 24px; color: #1f2937; }}
  h1    {{ font-size: 18px; margin-bottom: 4px; }}
  h2    {{ font-size: 15px; margin: 20px 0 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th    {{ background: #e5e7eb; padding: 4px 8px; text-align: left; }}
  .doc  {{ white-space: pre-wrap; line-height: 1.8; background: #fafafa;
           border: 1px solid #e5e7eb; padding: 16px; border-radius: 4px; }}
  .warn {{ color: #dc2626; font-weight: bold; }}
</style>
</head>
<body>
<h1>{doc_id}</h1>
{f'<p class="warn">⚠ {missing} entity(ies) not found in document</p>' if missing else ''}
{legend_html()}

<h2>Ground truth coverage</h2>
<table>
<thead><tr>
  <th>Entity text</th><th>Label</th><th>Propose</th><th>Hits</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>

<h2>Document</h2>
<div class="doc">{annotated}</div>
</body>
</html>"""


def most_recent_doc() -> Path | None:
    dirs = sorted(
        (d for d in OUTPUT_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return dirs[0] if dirs else None


def most_recent_schema() -> Path | None:
    if not SCHEMA_DIR.exists():
        return None
    files = sorted(
        (p for p in SCHEMA_DIR.iterdir() if p.is_file() and p.suffix == ".json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def spaced_positions(count: int, x: int, top: int, bottom: int) -> list[tuple[int, float]]:
    if count <= 0:
        return []
    if count == 1:
        return [(x, (top + bottom) / 2)]
    step = (bottom - top) / (count - 1)
    return [(x, top + idx * step) for idx in range(count)]


def compute_graph_layout(schema: dict) -> tuple[dict[str, tuple[int, float]], int, int]:
    persons = schema.get("persons", [])
    orgs = schema.get("orgs", [])
    defendants = [p for p in persons if p.get("type") == "defendant"]
    collateral = [p for p in persons if p.get("type") != "defendant"]
    charged = [o for o in orgs if o.get("type") == "charged"]
    associated = [o for o in orgs if o.get("type") != "charged"]

    max_group = max(
        1,
        len(defendants),
        len(collateral),
        len(charged),
        len(associated),
    )
    height = max(860, 180 + max_group * 92)
    width = 1680
    top = 140
    bottom = height - 120

    positions: dict[str, tuple[int, float]] = {}
    columns = [
        (defendants, 220),
        (collateral, 560),
        (charged, 1120),
        (associated, 1460),
    ]
    for nodes, x in columns:
        for node, (nx, ny) in zip(nodes, spaced_positions(len(nodes), x, top, bottom)):
            positions[node["id"]] = (nx, ny)

    return positions, width, height


def edge_line(
    source: tuple[int, float],
    target: tuple[int, float],
    color: str,
) -> str:
    sx, sy = source
    tx, ty = target
    dx = tx - sx
    dy = ty - sy
    distance = math.hypot(dx, dy) or 1.0
    trim = 34
    start_x = sx + dx * trim / distance
    start_y = sy + dy * trim / distance
    end_x = tx - dx * trim / distance
    end_y = ty - dy * trim / distance
    return (
        f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" '
        f'x2="{end_x:.1f}" y2="{end_y:.1f}" '
        f'stroke="{color}" stroke-width="2.4" marker-end="url(#arrow-{color[1:]})" />'
    )


def graph_html(schema_path: Path, schema: dict) -> str:
    positions, width, height = compute_graph_layout(schema)
    node_lookup = {
        **{node["id"]: node for node in schema.get("persons", [])},
        **{node["id"]: node for node in schema.get("orgs", [])},
    }
    people_count = len(schema.get("persons", []))
    org_count = len(schema.get("orgs", []))
    edge_count = len(schema.get("edges", []))

    arrow_defs = "".join(
        f'''
        <marker id="arrow-{color[1:]}" markerWidth="10" markerHeight="10"
                refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,7 L9,3.5 z" fill="{color}" />
        </marker>
        '''
        for color in sorted(set(EDGE_COLORS.values()))
    )

    edges_svg = []
    labels_svg = []
    for edge in schema.get("edges", []):
        source = positions.get(edge.get("from"))
        target = positions.get(edge.get("to"))
        if not source or not target:
            continue
        color = EDGE_COLORS.get(edge.get("type"), "#64748b")
        edges_svg.append(edge_line(source, target, color))

        sx, sy = source
        tx, ty = target
        mx = (sx + tx) / 2
        my = (sy + ty) / 2 - 10
        label = html.escape(edge.get("type", "edge").replace("_", " "))
        labels_svg.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" class="edge-label">{label}</text>'
        )

    nodes_svg = []
    for node_id, (x, y) in positions.items():
        node = node_lookup[node_id]
        node_type = node.get("type", "node")
        color = NODE_COLORS.get(node_type, "#e5e7eb")
        name = html.escape(node.get("display") or node.get("name") or node_id)
        raw_name = html.escape(node.get("name") or "")
        type_label = html.escape(node_type.replace("_", " "))
        nodes_svg.append(
            f'''
            <g class="node">
              <title>{raw_name} ({type_label})</title>
              <circle
                cx="{x}"
                cy="{y:.1f}"
                r="28"
                fill="{color}"
                stroke="#0f172a"
                stroke-width="1.5"
              />
              <text x="{x}" y="{y + 52:.1f}" class="node-label">{name}</text>
              <text x="{x}" y="{y + 70:.1f}" class="node-type">{type_label}</text>
            </g>
            '''
        )

    legend_items = "".join(
        (
            '<span class="legend-item">'
            f'<span class="swatch" style="background:{color}"></span>{label}'
            "</span>"
        )
        for label, color in (
            ("defendant", NODE_COLORS["defendant"]),
            ("collateral", NODE_COLORS["collateral"]),
            ("charged org", NODE_COLORS["charged"]),
            ("associated org", NODE_COLORS["associated"]),
        )
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(schema.get("doc_id", schema_path.stem))} graph</title>
<style>
  :root {{
    --bg: #f8fafc;
    --panel: #ffffff;
    --ink: #0f172a;
    --muted: #475569;
    --grid: #e2e8f0;
  }}
  body {{
    margin: 0;
    padding: 24px;
    background:
      radial-gradient(circle at top left, #e0f2fe 0, transparent 28%),
      radial-gradient(circle at top right, #fef3c7 0, transparent 26%),
      var(--bg);
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
  }}
  .panel {{
    max-width: 1720px;
    margin: 0 auto;
    background: var(--panel);
    border: 1px solid var(--grid);
    border-radius: 16px;
    box-shadow: 0 16px 50px rgba(15, 23, 42, 0.08);
    overflow: hidden;
  }}
  .header {{
    padding: 20px 24px 8px;
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 26px;
  }}
  p {{
    margin: 0;
    color: var(--muted);
    font-size: 15px;
  }}
  .legend {{
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    padding: 12px 24px 0;
    color: var(--muted);
    font-size: 14px;
  }}
  .legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }}
  .swatch {{
    width: 14px;
    height: 14px;
    border-radius: 999px;
    border: 1px solid rgba(15, 23, 42, 0.2);
  }}
  .graph-wrap {{
    overflow: auto;
    padding: 18px 18px 24px;
  }}
  svg {{
    width: 100%;
    min-width: {width}px;
    height: auto;
    background:
      linear-gradient(to right, rgba(226, 232, 240, 0.25) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(226, 232, 240, 0.25) 1px, transparent 1px),
      #fcfcfd;
    background-size: 48px 48px;
    border-radius: 14px;
  }}
  .zone-label {{
    font-size: 20px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    fill: #334155;
  }}
  .edge-label {{
    font-size: 12px;
    font-style: italic;
    text-anchor: middle;
    fill: #475569;
    paint-order: stroke;
    stroke: #ffffff;
    stroke-width: 5px;
    stroke-linejoin: round;
  }}
  .node-label {{
    text-anchor: middle;
    font-size: 12px;
    font-weight: bold;
    fill: #0f172a;
  }}
  .node-type {{
    text-anchor: middle;
    font-size: 11px;
    fill: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
</style>
</head>
<body>
  <div class="panel">
    <div class="header">
      <h1>{html.escape(schema.get("doc_id", schema_path.stem))}</h1>
      <p>
        {html.escape(schema.get("fraud_type", "unknown").replace("_", " "))}
        case graph with {people_count} people, {org_count} organisations, and
        {edge_count} edges.
      </p>
    </div>
    <div class="legend">{legend_items}</div>
    <div class="graph-wrap">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Case relationship graph">
        <defs>{arrow_defs}</defs>
        <text x="220" y="72" class="zone-label" text-anchor="middle">Defendants</text>
        <text x="560" y="72" class="zone-label" text-anchor="middle">Collateral</text>
        <text x="1120" y="72" class="zone-label" text-anchor="middle">Charged Orgs</text>
        <text x="1460" y="72" class="zone-label" text-anchor="middle">Associated Orgs</text>
        {''.join(edges_svg)}
        {''.join(labels_svg)}
        {''.join(nodes_svg)}
      </svg>
    </div>
  </div>
</body>
</html>"""


def process_graph(schema_path: Path) -> Path:
    schema = load_schema(schema_path)
    html_content = graph_html(schema_path, schema)
    out = schema_path.with_name(f"{schema_path.stem}_graph.html")
    out.write_text(html_content, encoding="utf-8")
    print(
        f"  {schema.get('doc_id', schema_path.stem)}: "
        f"{len(schema.get('edges', []))} edges → {out}"
    )
    return out


def process(doc_dir: Path) -> Path:
    doc_id = doc_dir.name
    text, entities = load_doc(doc_dir)
    html_content = render_html(doc_id, text, entities)
    out = doc_dir / f"{doc_id}.html"
    out.write_text(html_content, encoding="utf-8")
    hits   = sum(1 for e in entities if len(re.findall(re.escape(e[0]), text, re.IGNORECASE)) > 0)
    total  = len([e for e in entities if e[2] == "yes"])
    print(f"  {doc_id}: {hits}/{total} entities found  →  {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--doc", metavar="DOC_ID")
    group.add_argument("--graph", nargs="?", const="__latest__", metavar="DOC_ID")
    return parser


def resolve_graph_schema_path(graph_arg: str) -> Path:
    if graph_arg == "__latest__":
        schema_path = most_recent_schema()
        if schema_path is None:
            raise SystemExit("No schema files found in schemas/.")
        return schema_path

    schema_path = SCHEMA_DIR / f"{graph_arg}.json"
    if not schema_path.is_file():
        raise SystemExit(f"Schema file not found: {schema_path}")
    return schema_path


def open_hint(output_path: Path) -> None:
    print(f"\nOpen with:\n  open {output_path}")


def handle_graph_mode(graph_arg: str) -> None:
    output_path = process_graph(resolve_graph_schema_path(graph_arg))
    open_hint(output_path)


def iter_output_dirs() -> list[Path]:
    if not OUTPUT_DIR.exists():
        raise SystemExit("No output/ folder found. Run generator first.")
    return sorted(directory for directory in OUTPUT_DIR.iterdir() if directory.is_dir())


def resolve_doc_dir(doc_id: str | None) -> Path:
    if doc_id:
        doc_dir = OUTPUT_DIR / doc_id
        if not doc_dir.is_dir():
            raise SystemExit(f"Document folder not found: {doc_dir}")
        return doc_dir

    doc_dir = most_recent_doc()
    if doc_dir is None:
        raise SystemExit("No documents found in output/.")
    return doc_dir


def handle_document_mode(args: argparse.Namespace) -> None:
    if args.all:
        directories = iter_output_dirs()
        if not directories:
            raise SystemExit("No documents found in output/.")
        for directory in directories:
            process(directory)
        return

    output_path = process(resolve_doc_dir(args.doc))
    open_hint(output_path)


def main():
    args = build_parser().parse_args()

    if args.graph is not None:
        handle_graph_mode(args.graph)
        return

    handle_document_mode(args)


if __name__ == "__main__":
    main()
