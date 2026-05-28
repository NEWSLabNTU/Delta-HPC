"""
src/deploy/heuristic_agent.py
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
from src.bench.heuristic import RuleBasedHeuristic
from src.deploy.config import DEPLOY_CONFIG
from src.bench.config import BENCH_CONFIG


logger = logging.getLogger(__name__)


class _FakeAgent:
    def __init__(self, agent_id: m.AgentId):
        self.agent_id = agent_id
        self.engines = []


class _FakeEngine:
    def __init__(
        self,
        owner: _FakeAgent,
        mig_profile: m.MIGProfileBase,
        gpu_id: int,
        status: m.EngineStatus,
    ):
        self.owner = owner
        self.mig_profile = mig_profile
        self.gpu = gpu_id
        self.status = status
        self.waiting_queue = []


class FakeSimulatorAdapter:
    def __init__(self, act_ctrl):
        self.act_ctrl = act_ctrl

    def get_state(self) -> m.EnvironmentStateData:
        return OBS_COLLECTOR.get_observation()

    def get_action_mask(self, ignore_cooldowns: bool = False):
        return self.act_ctrl.get_action_mask(ignore_cooldowns=True)

    @property
    def agents(self):
        return self._build_state()[0]

    @property
    def gpu_engines(self):
        return self._build_state()[1]

    def _build_state(self):
        agents_dict = {aid: _FakeAgent(aid) for aid in m.AgentId}
        gpu_engines_dict = {}
        for gpu_id, gpu in SYSTEM_STATE.gpus.items():
            engines = []
            for slot in gpu.slots:
                owner = agents_dict[slot.agent_id]
                if slot.is_draining:
                    status = m.EngineStatus.DRAINING
                elif not slot.is_ready:
                    status = m.EngineStatus.BOOTING
                else:
                    status = m.EngineStatus.ACTIVE
                eng = _FakeEngine(owner, slot.profile_placement.profile, gpu_id, status)
                # Populate fake waiting queue length for heuristic to read
                stats = OBS_COLLECTOR._agent_stats[slot.agent_id]
                idx = (
                    6
                    if gpu.is_simulated
                    else slot.profile_placement.profile.profile_type.value
                )
                try:
                    q_len = int(stats.history["queue_length"][0][idx])
                except (IndexError, KeyError):
                    q_len = 0
                eng.waiting_queue = [None] * q_len

                owner.engines.append(eng)
                engines.append(eng)
            gpu_engines_dict[gpu_id] = engines
        return agents_dict, gpu_engines_dict

    def map_to_action(self, action):
        return self.act_ctrl.map_to_action(action)


def deploy_get_service_rate(agent_id, mig_profile, gpu_id=0):
    is_sim = SYSTEM_STATE.gpus[gpu_id].is_simulated
    if is_sim:
        return BENCH_CONFIG.get_service_rate(agent_id, mig_profile, gpu_id)
    return DEPLOY_CONFIG.get_service_rate(agent_id, mig_profile, gpu_id)


class HeuristicAgent(BasePolicyAgent):
    def __init__(self, act_ctrl: ActionController):
        super().__init__(act_ctrl)
        self.heuristic = RuleBasedHeuristic(get_service_rate=deploy_get_service_rate)

    async def run_loop(self, duration_s: float) -> None:
        OBS_COLLECTOR.start_budget_refresh_loop()
        start = time.time()
        step = 0

        logger.info(
            "Heuristic control loop started (duration=%.0fs, interval=%.0fs).",
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
                logger.info("Heuristic control loop finished after %d steps.", step)
                break

            OBS_COLLECTOR.start_new_interval()

            chosen_action = self.heuristic.decide_action(sim_adapter)

            if chosen_action != m.ResourceManagerAction.NO_ACTION:
                concrete_action = self.act_ctrl.map_to_action(chosen_action)
                if concrete_action is not None:
                    await self.act_ctrl.execute_action(concrete_action)

            step += 1
