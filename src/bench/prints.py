import tabulate
from typing import Any, Dict, List, Union

import src.share.models as m
import src.simulation.utils as u
from src.bench.models import BenchMode


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

    # Dynamically build per-agent columns
    active_agents = list(m.AgentId)
    agent_metrics = {aid: format_metrics(results[aid.value]) for aid in active_agents}
    agent_labels = [aid.value for aid in active_agents]

    metric_rows = [
        "TTFT (P25/50/75/99)",
        "TPOT (P25/50/75)",
        "Avg Q",
        "S/M/T",
        "MIG Existence (%)",
        "Tokens by MIG (%)",
    ]
    table_data = [
        [label] + [agent_metrics[aid][i] for aid in active_agents]
        for i, label in enumerate(metric_rows)
    ]

    print("\n● Aggregate Metrics")
    print(
        tabulate.tabulate(
            table_data,
            headers=["Metric"] + agent_labels,
            tablefmt="fancy_outline",
            headersglobalalign="center",
        )
    )

    sorted_mig_profiles = sorted(
        m.MIGProfile, key=lambda x: x.idx if hasattr(x, "idx") else x.value
    )
    mig_names = [get_mig_name(prof) for prof in sorted_mig_profiles]

    print("\n● Tokens by MIG Profile per Workload (%)")
    print("Format: 7G | 4G | 3G | 2G | 1L | 1S")
    for aid in active_agents:
        print(f"\n[{aid.value}]")
        tok_headers = ["Workload"] + mig_names
        tok_data: List[List[str]] = []
        tok_by_pat = results[aid.value]["token_mig_by_pattern"]
        for pat, mig_dict in tok_by_pat.items():
            row = [pat] + [
                f"{mig_dict.get(prof, 0.0):.1f}%"
                for prof in sorted_mig_profiles
            ]
            tok_data.append(row)
        # Overall row
        overall = results[aid.value]["overall_token_mig_percentages"]
        tok_data.append(
            ["[overall]"] + [f"{overall.get(prof, 0.0):.1f}%" for prof in sorted_mig_profiles]
        )
        print(
            tabulate.tabulate(
                tok_data,
                headers=tok_headers,
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
            table_data.append([
                aid.value if i == 0 else "",
                p["pattern"],
                f"{p['avg_rate']:.3f}",
                f"{p['duration']:.3f}",
                f"{p['proportion']:.1f}%",
            ])

    print(
        tabulate.tabulate(
            table_data,
            headers=headers,
            tablefmt="fancy_outline",
            headersglobalalign="center",
        )
    )


def print_initial_state():
    # Display Initial State
    print("\n[Initial State]")
    state_info: List[List[str]] = []
    for e in u.SIM_CONFIG.initial_state:
        if e.get("is-unused", False):
            state_info.append([
                str(e["gpu"]),
                "Unused (Pad)",
                str(e["mig"]),
                " ",
            ])
        else:
            state_info.append([
                str(e["gpu"]),
                str(e["agent"]),
                str(e["mig"]),
                "✓" if e.get("is-permanent", False) else " ",
            ])
    print(
        tabulate.tabulate(
            state_info,
            headers=["GPU", "Agent", "MIG", "Perm"],
            tablefmt="fancy_outline",
            headersglobalalign="center",
        )
    )
