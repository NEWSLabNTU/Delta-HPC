from typing import Dict, List


import src.share.models as m
from src.training.config import TRAINING_CONFIG


def _compute_gpu_affinity_bonus(gpu_engines: Dict[int, List[m.LLMEngine]]) -> float:
    agent_sm_gpu_num: Dict[m.AgentId, Dict[int, int]] = {
        agent_id: {gpu: 0 for gpu in gpu_engines.keys()} for agent_id in m.AgentId
    }

    for gpu_id, engines in gpu_engines.items():
        for engine in engines:
            if not engine.is_permanent:
                agent_sm_gpu_num[engine.owner.agent_id][gpu_id] += (
                    engine.mig_profile.size
                )

    bonus = 0.0
    for gpu_stat in agent_sm_gpu_num.values():
        total_sm = sum(gpu_stat.values())
        if total_sm == 0:
            continue
        for n_sm in gpu_stat.values():
            bonus += (n_sm / total_sm) ** 2
    return bonus


def compute_reward(
    requests: Dict[m.AgentId, List[m.Request]],
    action: m.ResourceManagerAction,
    current_time: float,
    gpu_engines: Dict[int, List[m.LLMEngine]],
    epsilon: float = 1e-9,
) -> float:
    w_t = TRAINING_CONFIG.w("ttft")
    w_p = TRAINING_CONFIG.w("tpot")
    use_quality_bonus = TRAINING_CONFIG.use_quality_bonus

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
                    else TRAINING_CONFIG.qf_concrete(
                        req.serving_engine.mig_profile, agent_id
                    )
                )
            else:
                assert req.generated_tokens > 0
                ttft = req.first_token_time - req.arrival_time
                tpot = req.decode_time / req.generated_tokens
                q_j = TRAINING_CONFIG.qf_concrete(
                    req.serving_engine.mig_profile, agent_id
                )

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
    affinity_bonus = 0.0
    if TRAINING_CONFIG.use_affinity_bonus:
        affinity_bonus = (
            _compute_gpu_affinity_bonus(gpu_engines)
            * TRAINING_CONFIG.affinity_bonus_weight
        )

    total_reward = (
        -total_penalty - omega + quality_bonus + affinity_bonus
    ) * TRAINING_CONFIG.scaling
    total_reward = max(total_reward, TRAINING_CONFIG.clip_threshold)

    return total_reward
