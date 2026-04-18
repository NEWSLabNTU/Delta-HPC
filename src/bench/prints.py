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
    def format_metrics(metrics: Dict[str, Any]) -> List[str]:
        ttft_str = "/".join([f"{x:.3f}" for x in metrics["ttft_percentiles"]])
        tpot_str = "/".join([f"{x:.3f}" for x in metrics["tpot_quartiles"]])
        avg_q_str = f"{metrics['avg_waiting_queue']:.3f}"
        smt_str = f"{metrics['split_count']}/{metrics['merge_count']}/{metrics['transfer_count']}"

        mig_tokens: List[str] = []
        for pat in ["idle", "balanced", "busy"]:
            pat_dict = metrics["token_mig_percentages"].get(pat, {})
            pat_str_parts: List[str] = []
            for mig in sorted(pat_dict.keys(), key=lambda m: m.size, reverse=True):
                val = pat_dict[mig]
                if val > 0.05:
                    pat_str_parts.append(f"{mig.size}g: {val:.0f}%")
            if pat_str_parts:
                mig_tokens.append(f"[{pat}] " + ", ".join(pat_str_parts))
            else:
                mig_tokens.append(f"[{pat}] n/a")
        mig_str = "\n".join(mig_tokens)

        mig_existence: List[str] = []
        for mig in sorted(
            metrics["mig_existence_percentages"].keys(), key=lambda m: m.size
        ):
            val = metrics["mig_existence_percentages"][mig]
            mig_existence.append(f"{mig.size}g: {val:.1f}%")
        existence_str = "\n".join(mig_existence)

        return [ttft_str, tpot_str, avg_q_str, smt_str, mig_str, existence_str]

    coding_metrics = format_metrics(results[m.AgentId.CODING.value])
    rag_metrics = format_metrics(results[m.AgentId.RAG.value])

    table_data = [
        ["TTFT (P25/50/75/99)", coding_metrics[0], rag_metrics[0]],
        ["TPOT (P25/50/75)", coding_metrics[1], rag_metrics[1]],
        ["Avg Q", coding_metrics[2], rag_metrics[2]],
        ["S/M/T", coding_metrics[3], rag_metrics[3]],
        ["Tokens By MIG (%)", coding_metrics[4], rag_metrics[4]],
        ["MIG Existence (%)", coding_metrics[5], rag_metrics[5]],
    ]

    print("\n● Aggregate Metrics")
    print(
        tabulate.tabulate(
            table_data,
            headers=["Metric", "Coding Agent", "RAG Agent"],
            tablefmt="fancy_outline",
        )
    )


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


def print_matrix_metrics(
    matrix_results: Dict[
        m.InitialMIGCombination,
        Dict[m.InitialMIGCombination, Dict[str, Dict[str, Any]]],
    ],
):
    col_keys = list(list(matrix_results.values())[0].keys())
    headers = ["GPU 0 \\ GPU 1"] + [c.name for c in col_keys]

    for aid in [m.AgentId.CODING, m.AgentId.RAG]:
        table_data: List[List[str]] = []
        for row_key, col_dict in matrix_results.items():
            row_data = [row_key.name]
            for col_key in col_keys:
                results = col_dict[col_key]
                metrics = results[aid.value]

                ttft = metrics["ttft_percentiles"]
                p25, p50, p75, p99 = ttft[0], ttft[1], ttft[2], ttft[3]
                trans = metrics.get("transfer_count", 0)

                cell_str = f"{p25:.2f}/{p50:.2f}/{p75:.2f}/{p99:.2f}/{trans}"
                row_data.append(cell_str)

            table_data.append(row_data)

        print(f"\n● Phase 1 Performance Matrix ({aid.name} Agent)")
        print("Format: TTFT @ 25 / 50 / 75 / 99 / Transfers")
        print(tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_outline"))
