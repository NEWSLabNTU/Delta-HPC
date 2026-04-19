import tabulate
from typing import Any, Dict, List

import src.simulation.models as m
import src.simulation.utils as utils
from src.bench.models import BenchMode, Workload
from src.bench.config import BENCH_CONFIG
from src.training.models import TrainingPhase


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

        mig_existence: List[str] = []
        for mig in sorted(
            metrics["mig_existence_percentages"].keys(), key=lambda m: m.size
        ):
            val = metrics["mig_existence_percentages"][mig]
            mig_existence.append(f"{mig.size}g: {val:.1f}%")
        existence_str = "\n".join(mig_existence)

        overall_token_mig: List[str] = []
        for mig in sorted(
            metrics["overall_token_mig_percentages"].keys(),
            key=lambda m: m.size,
        ):
            val = metrics["overall_token_mig_percentages"][mig]
            overall_token_mig.append(f"{mig.size}g: {val:.1f}%")
        overall_token_str = "\n".join(overall_token_mig)

        return [
            ttft_str,
            tpot_str,
            avg_q_str,
            smt_str,
            existence_str,
            overall_token_str,
        ]

    coding_metrics = format_metrics(results[m.AgentId.CODING.value])
    rag_metrics = format_metrics(results[m.AgentId.RAG.value])

    table_data = [
        ["TTFT (P25/50/75/99)", coding_metrics[0], rag_metrics[0]],
        ["TPOT (P25/50/75)", coding_metrics[1], rag_metrics[1]],
        ["Avg Q", coding_metrics[2], rag_metrics[2]],
        ["S/M/T", coding_metrics[3], rag_metrics[3]],
        ["MIG Existence (%)", coding_metrics[4], rag_metrics[4]],
        ["Tokens by MIG (%)", coding_metrics[5], rag_metrics[5]],
    ]

    print("\n● Aggregate Metrics")
    print(
        tabulate.tabulate(
            table_data,
            headers=["Metric", "Coding Agent", "RAG Agent"],
            tablefmt="fancy_outline",
            headersglobalalign="center",
        )
    )

    patterns = [w.value for w in Workload if w != Workload.HYBRID]

    print("\n● Tokens by MIG Matrix (%)")
    print("Format: 7G | 4G | 3G | 2G | 1G")
    for aid in [m.AgentId.CODING, m.AgentId.RAG]:
        print(f"\n[{aid.name} Agent]")
        headers = ["Coding \\ RAG"] + patterns
        mat_data: List[List[str]] = []
        token_mig_percentages = results[aid.value]["token_mig_percentages"]
        for pat_c in patterns:
            row_data = [pat_c]
            for pat_r in patterns:
                pat_dict = token_mig_percentages[pat_c][pat_r]
                mig_vals: List[str] = []
                for mig in sorted(pat_dict.keys(), key=lambda x: x.size, reverse=True):
                    mig_vals.append(f"{pat_dict[mig]:3.0f}%")
                row_data.append(" | ".join(mig_vals))
            mat_data.append(row_data)
        print(
            tabulate.tabulate(
                mat_data,
                headers=headers,
                tablefmt="fancy_outline",
                stralign="right",
                headersglobalalign="center",
            )
        )

    print("\n● Joint Workload Occurrences (%)")
    occ_headers = ["Coding \\ RAG"] + patterns
    occ_data: List[List[str]] = []
    occurrences = results[m.AgentId.CODING.value]["joint_occurrences"]
    for pat_c in patterns:
        row_data = [pat_c]
        for pat_r in patterns:
            row_data.append(f"{occurrences[pat_c][pat_r]:3.0f}%")
        occ_data.append(row_data)
    print(
        tabulate.tabulate(
            occ_data,
            headers=occ_headers,
            tablefmt="fancy_outline",
            stralign="right",
            headersglobalalign="center",
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

    print(
        tabulate.tabulate(
            table_data,
            headers=headers,
            tablefmt="fancy_outline",
            headersglobalalign="center",
        )
    )


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
        print(
            tabulate.tabulate(
                table_data,
                headers=headers,
                tablefmt="fancy_outline",
                headersglobalalign="center",
            )
        )

def print_initial_state(init_mode: Any):
    # Display Initial State
    if BENCH_CONFIG.phase == TrainingPhase.PHASE_1:
        print(f"\n[Initial State] {init_mode}")
        return
    print("\n[Initial State]")
    state_info: List[List[str]] = [
        [
            str(e["gpu"]),
            str(e["agent"]),
            str(e["mig"]),
            "✓" if e.get("is-permanent", False) else " ",
        ]
        for e in utils.SIM_CONFIG.initial_state
    ]
    print(
        tabulate.tabulate(
            state_info,
            headers=["GPU", "Agent", "MIG", "Perm"],
            tablefmt="fancy_outline",
            headersglobalalign="center",
        )
    )
