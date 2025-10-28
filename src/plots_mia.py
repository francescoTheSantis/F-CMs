import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from typing import List


def plot_and_save_max_mia(out_json="mia_summary.json", show=False):
    """
    Plot the results of the Membership Inference Attack (MIA) from a JSON file.
    Args:
        out_json (str): Path to the output JSON file for summary.
        show (bool): Whether to display the plots interactively.
    """
    # Load the MIA results from the JSON file
    file_in  = Path("mia_results.json")   # uploaded file 
    out_json = Path(out_json)             # output file for summary
    
    with file_in.open() as fp:
        raw = json.load(fp)

    def _extract_series(entry):
        if isinstance(entry, dict):
            rounds = entry.get("rounds", [])
            values = entry.get("values", [])
        else:
            values = entry
            rounds = list(range(1, len(values) + 1))
        rounds = [int(r) for r in rounds]
        values = [float(v) for v in values]
        return rounds, values

    max_acc_worst_list: List[float] = []
    max_eps_worst_list: List[float] = []
    max_acc_mean_list: List[float] = []
    max_eps_mean_list: List[float] = []
    for attack_type in ["whitebox", "blackbox", "blackbox_shadow"]: #"blackbox_concept",
        acc_data = raw["accuracies"].get(attack_type, {})
        eps_data = raw["epsilons"].get(attack_type, {})

        client_ids = sorted(acc_data.keys(), key=int)
        n_clients = len(client_ids)
        if n_clients == 0:
            max_acc_worst_list.append(None)
            max_eps_worst_list.append(None)
            max_acc_mean_list.append(None)
            max_eps_mean_list.append(None)
            continue

        client_acc_series = {}
        client_eps_series = {}
        acc_maps = {}
        eps_maps = {}

        for cid in client_ids:
            acc_rounds, acc_values = _extract_series(acc_data[cid])
            eps_rounds, eps_values = _extract_series(eps_data.get(cid, {}))
            client_acc_series[cid] = (acc_rounds, acc_values)
            client_eps_series[cid] = (eps_rounds, eps_values)
            acc_maps[cid] = {r: v for r, v in zip(acc_rounds, acc_values)}
            eps_maps[cid] = {r: v for r, v in zip(eps_rounds, eps_values)}

        # ========== FIGURE 1 : per‑client =========================================
        ncols = math.ceil(math.sqrt(n_clients))
        nrows = math.ceil(n_clients / ncols)
        fig1, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                sharex=True)
        axes = axes.ravel()

        for i, (ax, cid) in enumerate(zip(axes, client_ids)):
            rounds_acc, acc_values = client_acc_series[cid]
            rounds_eps, eps_values = client_eps_series[cid]

            if acc_values:
                ax.plot(rounds_acc, acc_values, label="MIA accuracy")
            if eps_values:
                ax.plot(rounds_eps, eps_values, label="ε (privacy budget)")

            if acc_values:
                max_acc_idx = int(np.argmax(acc_values))
                ax.scatter(rounds_acc[max_acc_idx], acc_values[max_acc_idx], s=40, zorder=5)
                ax.annotate(f"{acc_values[max_acc_idx]:.2f}",
                            (rounds_acc[max_acc_idx], acc_values[max_acc_idx]),
                            textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
            if eps_values:
                max_eps_idx = int(np.argmax(eps_values))
                ax.scatter(rounds_eps[max_eps_idx], eps_values[max_eps_idx], s=40, zorder=5)
                ax.annotate(f"{eps_values[max_eps_idx]:.2f}",
                            (rounds_eps[max_eps_idx], eps_values[max_eps_idx]),
                            textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)

            ax.set_title(f"Client {cid}")
            ax.set_xlabel("Round")
            ax.set_ylabel("Value")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.3)

            # Only add legend to the last subplot
            if i == n_clients - 1:
                ax.legend()

        # hide empty subplots (if any)
        for ax in axes[n_clients:]:
            ax.axis("off")

        fig1.tight_layout()

        # ========== AGGREGATED SERIES ============================================
        acc_round_sets = [set(acc_maps[cid].keys()) for cid in client_ids if acc_maps[cid]]
        eps_round_sets = [set(eps_maps[cid].keys()) for cid in client_ids if eps_maps[cid]]
        if acc_round_sets and eps_round_sets:
            common_rounds = sorted(set.intersection(*acc_round_sets) & set.intersection(*eps_round_sets))
        else:
            common_rounds = []

        if common_rounds:
            acc_max_by_round = []
            eps_max_by_round = []
            acc_mean_by_round = []
            eps_mean_by_round = []
            for rnd in common_rounds:
                acc_values_r = [acc_maps[cid][rnd] for cid in client_ids]
                eps_values_r = [eps_maps[cid][rnd] for cid in client_ids]
                acc_max_by_round.append(max(acc_values_r))
                eps_max_by_round.append(max(eps_values_r))
                acc_mean_by_round.append(float(np.mean(acc_values_r)))
                eps_mean_by_round.append(float(np.mean(eps_values_r)))

            acc_max_by_round = np.array(acc_max_by_round)
            eps_max_by_round = np.array(eps_max_by_round)
            acc_mean_by_round = np.array(acc_mean_by_round)
            eps_mean_by_round = np.array(eps_mean_by_round)

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            # subplot 1 : worst‑case series
            ax1.plot(common_rounds, acc_max_by_round, label="Worst‑case MIA accuracy")
            ax1.plot(common_rounds, eps_max_by_round, label="Worst‑case ε")
            max_acc_worst = float(acc_max_by_round.max())
            max_eps_worst = float(eps_max_by_round.max())
            worst_acc_idx = int(acc_max_by_round.argmax())
            worst_eps_idx = int(eps_max_by_round.argmax())
            ax1.scatter(common_rounds[worst_acc_idx], max_acc_worst, s=40)
            ax1.scatter(common_rounds[worst_eps_idx], max_eps_worst, s=40)
            ax1.annotate(f"{max_acc_worst:.2f}",
                        (common_rounds[worst_acc_idx], max_acc_worst),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
            ax1.annotate(f"{max_eps_worst:.2f}",
                        (common_rounds[worst_eps_idx], max_eps_worst),
                        textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)
            ax1.set_title("Worst‑case across clients per round")
            ax1.grid(alpha=0.3)
            ax1.legend()

            # subplot 2 : mean series
            ax2.plot(common_rounds, acc_mean_by_round, label="Mean MIA accuracy")
            ax2.plot(common_rounds, eps_mean_by_round, label="Mean ε")
            max_acc_mean = float(acc_mean_by_round.max())
            max_eps_mean = float(eps_mean_by_round.max())
            mean_acc_idx = int(acc_mean_by_round.argmax())
            mean_eps_idx = int(eps_mean_by_round.argmax())
            ax2.scatter(common_rounds[mean_acc_idx], max_acc_mean, s=40)
            ax2.scatter(common_rounds[mean_eps_idx], max_eps_mean, s=40)
            ax2.annotate(f"{max_acc_mean:.2f}",
                        (common_rounds[mean_acc_idx], max_acc_mean),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
            ax2.annotate(f"{max_eps_mean:.2f}",
                        (common_rounds[mean_eps_idx], max_eps_mean),
                        textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)
            ax2.set_title("Mean across clients per round")
            ax2.set_xlabel("Round")
            ax2.grid(alpha=0.3)
            ax2.legend()

            fig2.tight_layout()
            fig2.savefig(f"mia_aggregated_{attack_type}.png")

            max_acc_worst_list.append(max_acc_worst)
            max_eps_worst_list.append(max_eps_worst)
            max_acc_mean_list.append(max_acc_mean)
            max_eps_mean_list.append(max_eps_mean)
        else:
            max_acc_worst_list.append(None)
            max_eps_worst_list.append(None)
            max_acc_mean_list.append(None)
            max_eps_mean_list.append(None)
            fig2 = None

        fig1.savefig(f"mia_per_client_{attack_type}.png")

        if show:
            plt.show()
            
    # ========== SAVE SUMMARY JSON ============================================
    def _safe_float(value):
        return float(value) if value is not None else None

    summary = {
        "whitebox": {
            "worst_case": {
                "max_mia_accuracy": _safe_float(max_acc_worst_list[0]),
                "max_epsilon": _safe_float(max_eps_worst_list[0]),
            },
            "mean_across_clients": {
                "max_mia_accuracy": _safe_float(max_acc_mean_list[0]),
                "max_epsilon": _safe_float(max_eps_mean_list[0]),
            },
        },
        "blackbox": {
            "worst_case": {
                "max_mia_accuracy": _safe_float(max_acc_worst_list[1]),
                "max_epsilon": _safe_float(max_eps_worst_list[1]),
            },
            "mean_across_clients": {
                "max_mia_accuracy": _safe_float(max_acc_mean_list[1]),
                "max_epsilon": _safe_float(max_eps_mean_list[1]),
            },
        },
        # "blackbox_concept": {
        #     "worst_case": {
        #         "max_mia_accuracy": float(max_acc_worst_list[2]),
        #         "max_epsilon": float(max_eps_worst_list[2]),
        #     },
        #     "mean_across_clients": {
        #         "max_mia_accuracy": float(max_acc_mean_list[2]),
        #         "max_epsilon": float(max_eps_mean_list[2]),
        #     },
        # },
        "blackbox_shadow": {
            "worst_case": {
                "max_mia_accuracy": _safe_float(max_acc_worst_list[2]),
                "max_epsilon": _safe_float(max_eps_worst_list[2]),
            },
            "mean_across_clients": {
                "max_mia_accuracy": _safe_float(max_acc_mean_list[2]),
                "max_epsilon": _safe_float(max_eps_mean_list[2]),
            },
        },
    }
    with out_json.open("w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Summary JSON saved to {out_json}")


def plot_and_save_max_sia(out_json="sia_max.json", show=False) -> float:
    """
    Plot SIA accuracies over rounds and write the max value to a JSON file.

    Parameters
    ----------
    out_json : str or Path, default "sia_max.json"
        Destination JSON file with {"max_sia_accuracy": value}
    show : bool, default False
        Whether to display the plot interactively.

    Returns
    -------
    float
        The maximum SIA accuracy
    """
    json_path     = Path("sia_results.json")
    out_json_path = Path(out_json)

    with json_path.open() as fp:
        data = json.load(fp)
    accuracies = data["accuracies"]

    rounds = np.arange(1, len(accuracies) + 1)

    # ---------- plot -------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, accuracies, marker="o", label="SIA accuracy")
    max_idx = int(np.argmax(accuracies))
    max_val = float(accuracies[max_idx])
    plt.scatter(rounds[max_idx], max_val, s=60, zorder=5, label=f"max = {max_val:.3f}")
    plt.annotate(f"{max_val:.3f}", (rounds[max_idx], max_val),
                 textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("SIA accuracy per round")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sia_accuracy_plot.png")
    if show:
        plt.show()

    # ---------- save JSON --------------------------------------------------
    with out_json_path.open("w") as fp:
        json.dump({"max_sia_accuracy": max_val}, fp, indent=2)

    print(f"Max SIA accuracy {max_val:.4f} saved to {out_json_path}")
    return max_val




if __name__ == "__main__":
    plot_and_save_max_mia()
    plot_and_save_max_sia()
