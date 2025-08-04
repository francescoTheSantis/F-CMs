import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


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

    max_acc_worst_list = []
    max_eps_worst_list = []
    max_acc_mean_list = []
    max_eps_mean_list = []
    for attack_type in ["whitebox", "blackbox"]:
        acc_data = raw["accuracies"][attack_type]            # dict[str -> list]
        eps_data = raw["epsilons"][attack_type]

        client_ids = sorted(acc_data.keys(), key=int)
        n_clients  = len(client_ids)
        n_rounds   = len(next(iter(acc_data.values())))
        rounds     = np.arange(1, n_rounds + 1)

        # ========== FIGURE 1 : per‑client =========================================
        ncols = math.ceil(math.sqrt(n_clients))
        nrows = math.ceil(n_clients / ncols)
        fig1, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                sharex=True)
        axes = axes.ravel()

        for i, (ax, cid) in enumerate(zip(axes, client_ids)):
            acc = np.array(acc_data[cid])
            eps = np.array(eps_data[cid])

            line1, = ax.plot(rounds, acc, label="MIA accuracy")
            line2, = ax.plot(rounds, eps, label="ε (privacy budget)")

            # annotate maxima
            max_acc_idx = acc.argmax()
            max_eps_idx = eps.argmax()
            ax.scatter(rounds[max_acc_idx], acc[max_acc_idx], s=40, zorder=5)
            ax.scatter(rounds[max_eps_idx], eps[max_eps_idx], s=40, zorder=5)
            ax.annotate(f"{acc[max_acc_idx]:.2f}",
                        (rounds[max_acc_idx], acc[max_acc_idx]),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
            ax.annotate(f"{eps[max_eps_idx]:.2f}",
                        (rounds[max_eps_idx], eps[max_eps_idx]),
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
        acc_max_by_round = np.maximum.reduce([acc_data[cid] for cid in client_ids])
        eps_max_by_round = np.maximum.reduce([eps_data[cid] for cid in client_ids])

        acc_mean_by_round = np.mean([acc_data[cid] for cid in client_ids], axis=0)
        eps_mean_by_round = np.mean([eps_data[cid] for cid in client_ids], axis=0)

        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # subplot 1 : worst‑case series
        ax1.plot(rounds, acc_max_by_round, label="Worst‑case MIA accuracy")
        ax1.plot(rounds, eps_max_by_round, label="Worst‑case ε")
        max_acc_worst = acc_max_by_round.max()
        max_eps_worst = eps_max_by_round.max()
        ax1.scatter(acc_max_by_round.argmax()+1, max_acc_worst, s=40)
        ax1.scatter(eps_max_by_round.argmax()+1, max_eps_worst, s=40)
        ax1.annotate(f"{max_acc_worst:.2f}",
                    (acc_max_by_round.argmax()+1, max_acc_worst),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        ax1.annotate(f"{max_eps_worst:.2f}",
                    (eps_max_by_round.argmax()+1, max_eps_worst),
                    textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)
        ax1.set_title("Worst‑case across clients per round")
        ax1.grid(alpha=0.3)
        ax1.legend()

        # subplot 2 : mean series
        ax2.plot(rounds, acc_mean_by_round, label="Mean MIA accuracy")
        ax2.plot(rounds, eps_mean_by_round, label="Mean ε")
        max_acc_mean = acc_mean_by_round.max()
        max_eps_mean = eps_mean_by_round.max()
        ax2.scatter(acc_mean_by_round.argmax()+1, max_acc_mean, s=40)
        ax2.scatter(eps_mean_by_round.argmax()+1, max_eps_mean, s=40)
        ax2.annotate(f"{max_acc_mean:.2f}",
                    (acc_mean_by_round.argmax()+1, max_acc_mean),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        ax2.annotate(f"{max_eps_mean:.2f}",
                    (eps_mean_by_round.argmax()+1, max_eps_mean),
                    textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)
        ax2.set_title("Mean across clients per round")
        ax2.set_xlabel("Round")
        ax2.grid(alpha=0.3)
        ax2.legend()

        fig2.tight_layout()

        # ========== SAVE FIGURES ================================================
        fig1.savefig(f"mia_per_client_{attack_type}.png")
        fig2.savefig(f"mia_aggregated_{attack_type}.png")
        if show:
            plt.show()
            
        max_acc_worst_list.append(max_acc_worst)
        max_eps_worst_list.append(max_eps_worst)
        max_acc_mean_list.append(max_acc_mean)
        max_eps_mean_list.append(max_eps_mean)


    # ========== SAVE SUMMARY JSON ============================================
    summary = {
        "whitebox": {
            "worst_case": {
                "max_mia_accuracy": float(max_acc_worst_list[0]),
                "max_epsilon": float(max_eps_worst_list[0]),
            },
            "mean_across_clients": {
                "max_mia_accuracy": float(max_acc_mean_list[0]),
                "max_epsilon": float(max_eps_mean_list[0]),
            },
        },
        "blackbox": {
            "worst_case": {
                "max_mia_accuracy": float(max_acc_worst_list[1]),
                "max_epsilon": float(max_eps_worst_list[1]),
            },
            "mean_across_clients": {
                "max_mia_accuracy": float(max_acc_mean_list[1]),
                "max_epsilon": float(max_eps_mean_list[1]),
            },
        }
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
