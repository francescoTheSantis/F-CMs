import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import shutil

# ==========================
# CONFIG
# ==========================
XLSX_PATH = "temp_table.xlsx"  # <-- your new multi-sheet file
OUT_PDF = "neurips_plot_all_datasets.pdf"
OUT_PNG = "neurips_plot_all_datasets.png"

METHODS_ORDER = ["OURS", "FL", "LC"]

METHOD_STYLES = {
    "OURS": {"marker": "o", "linestyle": "-",  "linewidth": 1.8},
    "FL":   {"marker": "s", "linestyle": "--", "linewidth": 1.6},
    "LC":   {"marker": "^", "linestyle": ":",  "linewidth": 1.6},
}

def setup_icml_plot(two_column=True):
    """ICML/NeurIPS-like typography + seaborn styling."""
    figure_width = 7 if two_column else 3.5
    use_tex = shutil.which("latex") is not None
    if not use_tex:
        print("[warn] LaTeX not found; falling back to Matplotlib text rendering.")
    rc = {
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "figure.figsize": (figure_width, figure_width * 0.18),
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    sns.set_theme(context="paper", style="whitegrid", palette="deep", rc=rc)
    return figure_width


PAPER_WIDTH = setup_icml_plot(two_column=True)
METHOD_PALETTE = sns.color_palette("deep", n_colors=len(METHODS_ORDER))
METHOD_COLORS = dict(zip(METHODS_ORDER, METHOD_PALETTE))


# ==========================
# PARSER (SINGLE SHEET)
# ==========================
def parse_results_sheet(ws):
    """
    Parses one sheet with the same structure as your previous table.
    Returns a dataframe with columns: gap_prob, method, client_alt, mean, std
    """

    # 1) find where the header "% Client Alteration" is
    header_row, header_col = None, None
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            if ws.cell(r, c).value == "% Client Alteration":
                header_row, header_col = r, c
                break
        if header_row is not None:
            break

    if header_row is None:
        raise RuntimeError(f"[{ws.title}] Could not find '% Client Alteration' header.")

    # 2) read x-values (0.1 ... 0.9) from that header row
    client_alts = []
    mean_cols = []
    for c in range(header_col, ws.max_column + 1):
        v = ws.cell(header_row, c).value
        if isinstance(v, (int, float)) and v != 0:
            client_alts.append(float(v))
            mean_cols.append(c)

    if len(client_alts) == 0:
        raise RuntimeError(f"[{ws.title}] No client alteration values found (expected 0.1..0.9).")

    # 3) find where the data block starts ("Graph Alteration Probability")
    start_row = None
    for r in range(header_row, ws.max_row + 1):
        if ws.cell(r, header_col).value == "Graph Alteration Probability":
            start_row = r
            break

    if start_row is None:
        raise RuntimeError(f"[{ws.title}] Could not find 'Graph Alteration Probability' row.")

    # 4) parse rows
    records = []
    current_gap = None

    # Expected structure:
    # header_col     : "Graph Alteration Probability"
    # header_col + 1 : gap value (0.3, 0.6, 0.9, ...) often repeated sparsely
    # header_col + 2 : method name (OURS / FL / LC)
    # then alternating columns: mean, std for each client_alt
    for r in range(start_row, ws.max_row + 1):
        gap_val = ws.cell(r, header_col + 1).value
        method = ws.cell(r, header_col + 2).value

        if gap_val is not None:
            current_gap = float(gap_val)

        # stop heuristic: if first three block cells are empty -> likely end of table
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
                "dataset": ws.title,
                "gap_prob": current_gap,
                "method": method,
                "client_alt": float(alt),
                "mean": float(mean),
                "std": float(std) if std is not None else np.nan,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"[{ws.title}] Parsed dataframe is empty (no results detected).")

    return df


# ==========================
# READ ALL SHEETS
# ==========================
wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)

all_dfs = []
for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    df_sheet = parse_results_sheet(ws)
    all_dfs.append(df_sheet)

df_all = pd.concat(all_dfs, ignore_index=True)

datasets = list(df_all["dataset"].unique())
gap_order = sorted(df_all["gap_prob"].dropna().unique())

if len(gap_order) == 0:
    raise RuntimeError("No gap probabilities found across sheets.")


# ==========================
# PLOT (N rows x 3 cols)
# ==========================
n_rows = len(datasets)
n_cols = len(gap_order)

# A few ticks only -> paper-friendly
tick_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

# Figure size: keep width NeurIPS-like, height scales with datasets
fig_w = PAPER_WIDTH
fig_h = max(2.2, 1.9 * n_rows)  # 1.9 per dataset row

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(fig_w, fig_h),
    sharex=True,
    sharey="row",  # each dataset has its own y-scale, but columns match inside that dataset
)

# Handle edge cases (1 row or 1 col)
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = np.array([axes])
elif n_cols == 1:
    axes = np.array([[ax] for ax in axes])

for i, dataset in enumerate(datasets):
    df_d = df_all[df_all["dataset"] == dataset]

    # compute y-limits per row for clean comparison inside each dataset
    ymin = (df_d["mean"] - df_d["std"]).min()
    ymax = (df_d["mean"] + df_d["std"]).max()
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)

    for j, gap in enumerate(gap_order):
        ax = axes[i, j]
        sub = df_d[df_d["gap_prob"] == gap]

        # plot methods
        for method in METHODS_ORDER:
            d = sub[sub["method"] == method].sort_values("client_alt")
            if d.empty:
                continue

            x = d["client_alt"].values
            y = d["mean"].values
            s = d["std"].values

            sns.lineplot(
                data=d,
                x="client_alt",
                y="mean",
                ax=ax,
                label=method,
                color=METHOD_COLORS[method],
                marker=METHOD_STYLES[method]["marker"],
                linestyle=METHOD_STYLES[method]["linestyle"],
                linewidth=METHOD_STYLES[method]["linewidth"],
                markersize=4,
            )
            ax.fill_between(
                x,
                y - s,
                y + s,
                alpha=0.15,
                color=METHOD_COLORS[method],
                linewidth=0,
            )

        # Column titles only once (top row)
        if i == 0:
            ax.set_title(f"Graph alteration p={gap:g}", pad=8)

        # Dataset label + y-label on left-most subplot of each row
        if j == 0:
            ax.set_ylabel(f"{dataset}\nDiff. pairs (↓)")
        else:
            ax.set_ylabel("")

        # X label only on bottom row
        if i == n_rows - 1:
            ax.set_xlabel("% client alteration")
        else:
            ax.set_xlabel("")

        ax.set_xticks(tick_vals)
        ax.set_xlim(0.08, 0.92)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, linewidth=0.4, alpha=0.3, linestyle="--")

# Global legend (one only)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

sns.despine(fig=fig)
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

print(f"[OK] Saved: {OUT_PDF}")
print(f"[OK] Saved: {OUT_PNG}")
