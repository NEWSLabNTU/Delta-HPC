import tabulate
from typing import Any, Dict, List

import src.simulation.models as m
from src.bench.models import BenchMode


def print_banner(mode: BenchMode, running_id: str):
    width = 60

    def print_row(text: str):
        n_space = width - 2 - len(text)
        half = n_space // 2
        left = half
        right = half if n_space % 2 == 0 else half + 1
        print("#" + " " * left + text + " " * right + "#")

    print()
    print("#" * width)
    print_row(mode.name)
    if running_id:
        print_row(running_id)
    print("#" * width)
    print()


def print_metrics(results: Dict[str, Any]):
    for aid, metrics in results.items():
        ttft_str = "/".join([f"{x:.3f}" for x in metrics["ttft_percentiles"]])
        tpot_str = "/".join([f"{x:.3f}" for x in metrics["tpot_quartiles"]])
        avg_q_str = f"{metrics['avg_waiting_queue']:.3f}"
        smt_str = f"{metrics['split_count']}/{metrics['merge_count']}/{metrics['transfer_count']}"

        mig_tokens: List[str] = []
        for mig in sorted(
            metrics["token_mig_percentages"].keys(), key=lambda m: m.size
        ):
            val = metrics["token_mig_percentages"][mig]
            mig_tokens.append(f"{mig.size}g: {val:.1f}%")
        mig_str = "\n".join(mig_tokens)

        mig_existence: List[str] = []
        for mig in sorted(
            metrics["mig_existence_percentages"].keys(), key=lambda m: m.size
        ):
            val = metrics["mig_existence_percentages"][mig]
            mig_existence.append(f"{mig.size}g: {val:.1f}%")
        existence_str = "\n".join(mig_existence)

        # Build vertical data
        table_data = [
            ["TTFT (P25/50/75/99)", ttft_str],
            ["TPOT (P25/50/75)", tpot_str],
            ["Avg Q", avg_q_str],
            ["S/M/T", smt_str],
            ["Tokens By MIG (%)", mig_str],
            ["MIG Existence (%)", existence_str],
        ]

        print(f"\n● Agent: {aid}")
        print(tabulate.tabulate(table_data, tablefmt="fancy_outline"))


def print_workloads(summary: Dict[m.AgentId, List[Dict[str, Any]]]):
    print("\nWorkload Summary (Actual Patterns Encountered)")
    headers = ["Agent", "Pattern", "Avg Rate (req/s)", "Duration (s)", "Proportion (%)"]
    table_data: List[str | List[str]] = []

    for j, aid in enumerate(sorted(summary.keys(), key=lambda a: a.value)):
        if j > 0:
            table_data.append(tabulate.SEPARATING_LINE)

        phases = summary[aid]
        for i, p in enumerate(phases):
            table_data.append(
                [
                    aid.value if i == 0 else "",
                    p["pattern"],
                    f"{p['avg_rate']:.3f}",
                    f"{p['duration']:.3f}",
                    f"{p['proportion']:.1f}%",
                ]
            )

    print(tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_outline"))
