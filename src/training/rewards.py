from typing import Dict, List

import src.simulation.models as m
from src.training.config import TRAINING_CONFIG


def compute_reward(
    requests: Dict[m.AgentId, List[m.Request]],
    action: m.ResourceManagerAction,
    current_time: float,
    epsilon: float = 1e-9,
) -> float:
    """
    Computes the RL agent's reward based on latency and reconfiguration cost.
    """
    gamma = TRAINING_CONFIG.gamma
    w_t = TRAINING_CONFIG.w("ttft")
    w_p = TRAINING_CONFIG.w("tpot")

    # Omega(a_t): Penalty for taking an action
    omega = gamma if action != m.ResourceManagerAction.NO_ACTION else 0.0

    total_penalty = 0.0

    for agent_id, agent_requests in requests.items():
        alpha_k = TRAINING_CONFIG.alpha(agent_id)

        sum_latency = 0.0
        count = 0

        for req in agent_requests:
            if req.serving_engine is None:
                continue
            count += 1

            if req.first_token_time is None:
                ttft = current_time - req.arrival_time
                tpot = 0.0
                q_j = TRAINING_CONFIG.default_waiting_qj
            else:
                assert req.generated_tokens > 0
                ttft = req.first_token_time - req.arrival_time
                tpot = req.decode_time / req.generated_tokens
                q_j = TRAINING_CONFIG.qf(req.serving_engine.mig_profile)

            composite_latency = (w_t * ttft + w_p * tpot) / q_j
            sum_latency += composite_latency

        psi_k = sum_latency / (count + epsilon)
        total_penalty += alpha_k * psi_k

    total_reward = -(total_penalty + omega)
    total_reward = max(total_reward, TRAINING_CONFIG.clip_threshold)

    return total_reward
