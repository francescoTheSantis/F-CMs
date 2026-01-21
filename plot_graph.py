import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

# # ==========================
# # CONFIG
# # ==========================
# XLSX_PATH = "temp_table.xlsx"  # <-- change if needed
# OUT_PDF = "neurips_plot_asia_diffpairs.pdf"
# OUT_PNG = "neurips_plot_asia_diffpairs.png"

# METHODS_ORDER = ["OURS", "FL", "LC"]

# METHOD_STYLES = {
#     "OURS": {"marker": "o", "linestyle": "-",  "linewidth": 1.8},
#     "FL":   {"marker": "s", "linestyle": "--", "linewidth": 1.6},
#     "LC":   {"marker": "^", "linestyle": ":",  "linewidth": 1.6},
# }

# # NeurIPS-ish font sizes (compact but readable)
# plt.rcParams.update({
#     "font.size": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 9,
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })


# # ==========================
# # PARSE THE XLSX TABLE
# # ==========================
# wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
# ws = wb.active

# # Find header row/col where "% Client Alteration" is located
# header_row, header_col = None, None
# for r in range(1, ws.max_row + 1):
#     for c in range(1, ws.max_column + 1):
#         if ws.cell(r, c).value == "% Client Alteration":
#             header_row, header_col = r, c
#             break
#     if header_row is not None:
#         break

# if header_row is None:
#     raise RuntimeError("Could not find '% Client Alteration' in the spreadsheet.")

# # Extract x-values (client alteration) and corresponding "mean" column indices
# client_alts = []
# mean_cols = []
# for c in range(header_col, ws.max_column + 1):
#     v = ws.cell(header_row, c).value
#     if isinstance(v, (int, float)) and v != 0:
#         client_alts.append(float(v))
#         mean_cols.append(c)

# if len(client_alts) == 0:
#     raise RuntimeError("No client alteration values found (expected 0.1, 0.2, ..., 0.9).")

# # Find start of the data block
# start_row = None
# for r in range(header_row, ws.max_row + 1):
#     if ws.cell(r, header_col).value == "Graph Alteration Probability":
#         start_row = r
#         break

# if start_row is None:
#     raise RuntimeError("Could not find 'Graph Alteration Probability' in the spreadsheet.")

# # Parse the blocks into a dataframe: (gap_prob, method, client_alt, mean, std)
# records = []
# current_gap = None

# # In your file:
# # header_col     -> "Graph Alteration Probability"
# # header_col + 1 -> gap value (0.3, 0.6, 0.9, ...)
# # header_col + 2 -> method (OURS, FL, LC)
# # mean/std start -> header_col + 3 onward, in pairs (mean, std)
# for r in range(start_row, ws.max_row + 1):
#     gap_val = ws.cell(r, header_col + 1).value
#     method = ws.cell(r, header_col + 2).value

#     # update gap when present
#     if gap_val is not None:
#         current_gap = float(gap_val)

#     # stop if we reached empty area
#     if method is None:
#         # heuristic: if the first three cells of the block are empty -> table ended
#         if all(ws.cell(r, cc).value is None for cc in range(header_col, header_col + 3)):
#             break
#         continue

#     method = str(method).strip()

#     for alt, c_mean in zip(client_alts, mean_cols):
#         mean = ws.cell(r, c_mean).value
#         std = ws.cell(r, c_mean + 1).value
#         if mean is None:
#             continue

#         records.append({
#             "gap_prob": current_gap,
#             "method": method,
#             "client_alt": float(alt),
#             "mean": float(mean),
#             "std": float(std) if std is not None else np.nan,
#         })

# df = pd.DataFrame(records)
# if df.empty:
#     raise RuntimeError("Parsed dataframe is empty: no results detected.")


# # ==========================
# # PLOT
# # ==========================
# gap_order = sorted(df["gap_prob"].unique())
# tick_vals = [0.1, 0.3, 0.5, 0.7, 0.9]  # fewer ticks -> cleaner on paper

# fig, axes = plt.subplots(1, len(gap_order), figsize=(6.6, 2.2), sharey=True)
# if len(gap_order) == 1:
#     axes = [axes]

# ymin = (df["mean"] - df["std"]).min()
# ymax = (df["mean"] + df["std"]).max()
# pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)

# for ax, gap in zip(axes, gap_order):
#     sub = df[df["gap_prob"] == gap]

#     for method in METHODS_ORDER:
#         d = sub[sub["method"] == method].sort_values("client_alt")
#         x = d["client_alt"].values
#         y = d["mean"].values
#         s = d["std"].values

#         ax.plot(
#             x, y,
#             label=method,
#             marker=METHOD_STYLES[method]["marker"],
#             linestyle=METHOD_STYLES[method]["linestyle"],
#             linewidth=METHOD_STYLES[method]["linewidth"],
#             markersize=4,
#         )

#         # Shaded std band (mean ± std)
#         ax.fill_between(x, y - s, y + s, alpha=0.15)

#     ax.set_title(f"Graph alteration p={gap:g}", pad=8)
#     ax.set_xlabel("% client alteration")
#     ax.set_xticks(tick_vals)
#     ax.set_xlim(0.08, 0.92)
#     ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.4)

# axes[0].set_ylabel("Diff. pairs (↓)")
# for ax in axes:
#     ax.set_ylim(ymin - pad, ymax + pad)

# # Single legend (top, 3 columns)
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)

# fig.tight_layout(rect=[0, 0, 1, 0.98])

# fig.savefig(OUT_PDF, bbox_inches="tight")
# fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

# print(f"[OK] Saved: {OUT_PDF}")
# print(f"[OK] Saved: {OUT_PNG}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from matplotlib.lines import Line2D

# ==========================
# CONFIG
# ==========================
XLSX_PATH = "temp_table.xlsx"  # <-- change if needed
OUT_PDF = "neurips_plot_allinone.pdf"
OUT_PNG = "neurips_plot_allinone.png"

METHODS_ORDER = ["OURS", "FL", "LC"]

# Marker encodes METHOD (fixed across all lines)
METHOD_STYLE = {
    "OURS": {"marker": "o", "linestyle": "-",  "linewidth": 1.8},
    "FL":   {"marker": "s", "linestyle": "--", "linewidth": 1.5},
    "LC":   {"marker": "^", "linestyle": ":",  "linewidth": 1.5},
}

# NeurIPS-friendly typography
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ==========================
# PARSE THE XLSX TABLE
# ==========================
wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
ws = wb.active

# Find "% Client Alteration" header
header_row, header_col = None, None
for r in range(1, ws.max_row + 1):
    for c in range(1, ws.max_column + 1):
        if ws.cell(r, c).value == "% Client Alteration":
            header_row, header_col = r, c
            break
    if header_row is not None:
        break

if header_row is None:
    raise RuntimeError("Could not find '% Client Alteration' in the spreadsheet.")

# Extract x-values (0.1..0.9) and their "mean" column positions
client_alts = []
mean_cols = []
for c in range(header_col, ws.max_column + 1):
    v = ws.cell(header_row, c).value
    if isinstance(v, (int, float)) and v != 0:
        client_alts.append(float(v))
        mean_cols.append(c)

if len(client_alts) == 0:
    raise RuntimeError("No client alteration values found (expected 0.1..0.9).")

# Find where the data starts
start_row = None
for r in range(header_row, ws.max_row + 1):
    if ws.cell(r, header_col).value == "Graph Alteration Probability":
        start_row = r
        break

if start_row is None:
    raise RuntimeError("Could not find 'Graph Alteration Probability' in the spreadsheet.")

records = []
current_gap = None

# Expected layout:
# col+1 -> gap_prob
# col+2 -> method
# from mean_cols onwards -> mean/std pairs
for r in range(start_row, ws.max_row + 1):
    gap_val = ws.cell(r, header_col + 1).value
    method = ws.cell(r, header_col + 2).value

    # gap updates when present
    if gap_val is not None:
        current_gap = float(gap_val)

    # stop at empty region
    if method is None:
        if all(ws.cell(r, cc).value is None for cc in range(header_col, header_col + 3)):
            break
        continue

    method = str(method).strip()

    for alt, c_mean in zip(client_alts, mean_cols):
        mean = ws.cell(r, c_mean).value
        std = ws.cell(r, c_mean + 1).value
        if mean is None:
            continue

        records.append({
            "gap_prob": current_gap,
            "method": method,
            "client_alt": float(alt),
            "mean": float(mean),
            "std": float(std) if std is not None else np.nan,
        })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("Parsed dataframe is empty: no results detected.")


# ==========================
# PLOT (single figure)
# ==========================
gap_order = sorted(df["gap_prob"].unique())

# Use a colormap so each gap_prob gets a distinct color
cmap = plt.get_cmap("tab10")
color_map = {gap: cmap(i % 10) for i, gap in enumerate(gap_order)}

fig, ax = plt.subplots(figsize=(6.8, 3.2))

for gap in gap_order:
    for method in METHODS_ORDER:
        sub = df[(df["gap_prob"] == gap) & (df["method"] == method)].sort_values("client_alt")
        if sub.empty:
            continue

        x = sub["client_alt"].values
        y = sub["mean"].values
        s = sub["std"].values
        col = color_map[gap]

        # Main line
        ax.plot(
            x, y,
            color=col,
            marker=METHOD_STYLE[method]["marker"],
            linestyle=METHOD_STYLE[method]["linestyle"],
            linewidth=METHOD_STYLE[method]["linewidth"],
            markersize=4,
            alpha=0.95
        )

        # Std band
        ax.fill_between(x, y - s, y + s, color=col, alpha=0.12, linewidth=0)


# Axes styling
ax.set_xlabel("% client alteration")
ax.set_ylabel("Diff. pairs (↓)")
ax.set_xlim(0.08, 0.92)
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.4)

# Optional: tight y-limits from data
ymin = (df["mean"] - df["std"]).min()
ymax = (df["mean"] + df["std"]).max()
pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
ax.set_ylim(ymin - pad, ymax + pad)


# ==========================
# TWO LEGENDS (method markers + gap colors)
# ==========================
# Legend 1: method (marker encodes method)
method_handles = [
    Line2D([0], [0],
           color="black",
           marker=METHOD_STYLE[m]["marker"],
           linestyle=METHOD_STYLE[m]["linestyle"],
           linewidth=1.6,
           markersize=5,
           label=m)
    for m in METHODS_ORDER
]

# Legend 2: graph alteration probability (color encodes gap_prob)
gap_handles = [
    Line2D([0], [0],
           color=color_map[g],
           linestyle="-",
           linewidth=2.2,
           label=f"p={g:g}")
    for g in gap_order
]

leg1 = ax.legend(handles=method_handles, title="Method", loc="upper right", frameon=True)
ax.add_artist(leg1)  # keep it when adding second legend

ax.legend(handles=gap_handles, title="Graph alteration", loc="lower left", frameon=True)


fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

print(f"[OK] Saved: {OUT_PDF}")
print(f"[OK] Saved: {OUT_PNG}")
