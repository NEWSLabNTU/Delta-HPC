"""
src/deploy/qas_agent.py
"""

import time
import asyncio
import logging

import src.share.models as m
from src.deploy.obs import OBS_COLLECTOR
from src.training.config import TRAINING_CONFIG
from src.deploy.system import SYSTEM_STATE
from src.deploy.act_controller import ActionController
from src.deploy.base_agent import BasePolicyAgent
from src.bench.qas import QualityAwareScheduler
from src.deploy.config import DEPLOY_CONFIG
from src.bench.config import BENCH_CONFIG
from src.deploy.heuristic_agent import FakeSimulatorAdapter, deploy_get_service_rate


logger = logging.getLogger(__name__)


def deploy_get_ttft(agent_id, mig_profile, gpu_id, lam):
    is_sim = SYSTEM_STATE.gpus[gpu_id].is_simulated
    if is_sim:
        return BENCH_CONFIG.predict_ttft(agent_id, mig_profile, gpu_id, lam)
    return DEPLOY_CONFIG.predict_ttft(agent_id, mig_profile, gpu_id, lam)


class QASAgent(BasePolicyAgent):
    def __init__(self, act_ctrl: ActionController):
        super().__init__(act_ctrl)
        self.qas = QualityAwareScheduler(
            get_service_rate=deploy_get_service_rate, get_ttft=deploy_get_ttft
        )

    async def run_loop(self, duration_s: float) -> None:
        OBS_COLLECTOR.start_budget_refresh_loop()
        start = time.time()
        step = 0

        logger.info(
            "QAS control loop started (duration=%.0fs, interval=%.0fs).",
            duration_s,
            TRAINING_CONFIG.action_interval,
        )

        sim_adapter = FakeSimulatorAdapter(self.act_ctrl)

        while True:
            # Sleep until the next action trigger point
            next_trigger = start + (step + 1) * TRAINING_CONFIG.action_interval
            sleep_for = next_trigger - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            if time.time() - start >= duration_s:
                logger.info("QAS control loop finished after %d steps.", step)
                break

            OBS_COLLECTOR.start_new_interval()

            chosen_action = self.qas.decide_action(sim_adapter)

            if chosen_action != m.ResourceManagerAction.NO_ACTION:
                concrete_action = self.act_ctrl.map_to_action(chosen_action)
                if concrete_action is not None:
                    await self.act_ctrl.execute_action(concrete_action)

            step += 1
