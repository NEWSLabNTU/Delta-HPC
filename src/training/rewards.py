from typing import Dict, List, Optional

import src.simulation.models as m
from src.training.config import TRAINING_CONFIG


def _compute_gpu_affinity_bonus(agents: Dict[m.AgentId, m.Agent]) -> float:
    """
    Computes GPU affinity bonus: Is_Pure(k) * bonus for each non-permanent GPU.

    Is_Pure(k) = 1 if GPU k is exclusively owned by one agent (all its active
    MIG engines belong to a single agent), 0 otherwise. GPU 2 is excluded.
    """
    bonus_per_pure_gpu = TRAINING_CONFIG.gpu_affinity_bonus

    # Collect the set of agent owners per GPU (excluding GPU 2)
    gpu_owners: Dict[int, set] = {}
    for agent_id, agent in agents.items():
        for e in agent.engines:
            if e.is_permanent:
                continue  # skip permanent GPU
            gpu = e.gpu
            if gpu not in gpu_owners:
                gpu_owners[gpu] = set()
            gpu_owners[gpu].add(agent_id)

    affinity_bonus = 0.0
    for owners in gpu_owners.values():
        if len(owners) == 1:  # GPU is pure — owned by exactly one agent
            affinity_bonus += bonus_per_pure_gpu

    return affinity_bonus


def compute_reward(
    requests: Dict[m.AgentId, List[m.Request]],
    action: m.ResourceManagerAction,
    current_time: float,
    agents: Optional[Dict[m.AgentId, m.Agent]] = None,
    epsilon: float = 1e-9,
) -> float:
    w_t = TRAINING_CONFIG.w("ttft")
    w_p = TRAINING_CONFIG.w("tpot")
    use_quality_bonus = TRAINING_CONFIG.quality_bonus

    # Omega(a_t): Penalty for taking an action
    omega = (
        TRAINING_CONFIG.gamma if action != m.ResourceManagerAction.NO_ACTION else 0.0
    )

    total_penalty = 0.0

    # Accumulators for the quality bonus (across all agents)
    total_tokens = 0.0
    weighted_tokens = 0.0

    for agent_id, agent_requests in requests.items():
        alpha_k = TRAINING_CONFIG.alpha(agent_id)

        sum_latency = 0.0
        req_count = 0

        for req in agent_requests:
            if req.serving_engine is None:
                continue
            req_count += 1
            req_tokens = req.prompt_tokens + req.completion_tokens
            processed_tokens = req.prefilled_tokens + req.generated_tokens

            if req.first_token_time is None:  # still waiting for first token
                ttft = current_time - req.arrival_time
                tpot = 0.0
                q_j = (
                    TRAINING_CONFIG.default_waiting_qj
                    if req.prefilled_tokens == 0
                    else TRAINING_CONFIG.qf(req.serving_engine.mig_profile)
                )
            else:
                assert req.generated_tokens > 0
                ttft = req.first_token_time - req.arrival_time
                tpot = req.decode_time / req.generated_tokens
                q_j = TRAINING_CONFIG.qf(req.serving_engine.mig_profile)

            if use_quality_bonus:
                # Latency without Qf denominator
                composite_latency = w_t * ttft + w_p * tpot
                # Accumulate bonus terms
                total_tokens += req_tokens
                weighted_tokens += q_j * processed_tokens
            else:
                composite_latency = (w_t * ttft + w_p * tpot) / q_j

            sum_latency += composite_latency

        psi_k = sum_latency / (req_count + epsilon)
        total_penalty += alpha_k * psi_k

    # Quality bonus: ratio of Qf-weighted tokens to total tokens
    if use_quality_bonus:
        quality_bonus = weighted_tokens / (total_tokens + epsilon)
    else:
        quality_bonus = 0.0

    # GPU affinity bonus
    if TRAINING_CONFIG.gpu_affinity and agents is not None:
        affinity_bonus = _compute_gpu_affinity_bonus(agents)
    else:
        affinity_bonus = 0.0

    total_reward = (
        -total_penalty - omega + quality_bonus + affinity_bonus
    ) * TRAINING_CONFIG.scaling
    total_reward = max(total_reward, TRAINING_CONFIG.clip_threshold)

    return total_reward
