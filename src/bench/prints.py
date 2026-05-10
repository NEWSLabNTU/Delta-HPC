import tabulate
from typing import Any, Dict, List, Union

import src.simulation.models as m
import src.simulation.utils as utils
from src.bench.models import BenchMode, Workload


def get_mig_name(mig: Union[m.MIGProfile, m.MIGProfileBase]) -> str:
    profile = mig.profile_type if isinstance(mig, m.MIGProfileBase) else mig
    mapping = {
        m.MIGProfile.MIG_7G: "7G",
        m.MIGProfile.MIG_4G: "4G",
        m.MIGProfile.MIG_3G: "3G",
        m.MIGProfile.MIG_2G: "2G",
        m.MIGProfile.MIG_1G_LARGE: "1L",
        m.MIGProfile.MIG_1G_SMALL: "1S",
    }
    return mapping[profile]


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
        # Sort by logical profile index (7G=0, ..., 1S=5)
        sorted_migs = sorted(
            metrics["mig_existence_percentages"].keys(),
            key=lambda x: x.idx if hasattr(x, "idx") else x.value,
        )
        for mig in sorted_migs:
            val = metrics["mig_existence_percentages"][mig]
            name = get_mig_name(mig)
            mig_existence.append(f"{name}: {val:.1f}%")
        existence_str = "\n".join(mig_existence)

        overall_token_mig: List[str] = []
        sorted_token_migs = sorted(
            metrics["overall_token_mig_percentages"].keys(),
            key=lambda x: x.idx if hasattr(x, "idx") else x.value,
        )
        for mig in sorted_token_migs:
            val = metrics["overall_token_mig_percentages"][mig]
            name = get_mig_name(mig)
            overall_token_mig.append(f"{name}: {val:.1f}%")
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
    print("Format: 7G | 4G | 3G | 2G | 1L | 1S")
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
                # Sort by logical profile index
                sorted_pat_migs = sorted(
                    pat_dict.keys(),
                    key=lambda x: x.idx if hasattr(x, "idx") else x.value,
                )
                for mig in sorted_pat_migs:
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


def print_initial_state(init_mode: Any):
    # Display Initial State
    print("\n[Initial State]")
    state_info: List[List[str]] = []
    for e in utils.SIM_CONFIG.initial_state:
        if e.get("is-unused", False):
            state_info.append(
                [
                    str(e["gpu"]),
                    "Unused (Pad)",
                    str(e["mig"]),
                    " ",
                ]
            )
        else:
            state_info.append(
                [
                    str(e["gpu"]),
                    str(e["agent"]),
                    str(e["mig"]),
                    "✓" if e.get("is-permanent", False) else " ",
                ]
            )
    print(
        tabulate.tabulate(
            state_info,
            headers=["GPU", "Agent", "MIG", "Perm"],
            tablefmt="fancy_outline",
            headersglobalalign="center",
        )
    )
