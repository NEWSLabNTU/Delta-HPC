"""
src/deploy/rl.py

Wraps a trained MaskablePPO checkpoint for inference during live deployment.

The model is deliberately loaded onto the CPU so that all GPU memory remains
available for vLLM containers.  Observation construction mirrors
:meth:`src.share.env.BaseMIGResourceEnv._get_obs` exactly so that the policy
sees the same feature vector it was trained on.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, cast, Dict

import torch
import numpy as np
import numpy.typing as npt
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

import src.share.models as m
from src.deploy.obs import OBS_COLLECTOR
from src.deploy.base_agent import BasePolicyAgent
from src.training.config import TRAINING_CONFIG

if TYPE_CHECKING:
    from src.deploy.act_controller import ActionController

logger = logging.getLogger(__name__)


class RLAgent(BasePolicyAgent):
    """Loads a MaskablePPO checkpoint and runs inference on CPU.

    Parameters
    ----------
    ckpt_path:
        Path to the ``.zip`` checkpoint saved by ``MaskablePPO.save()``.
    vecnorm_path:
        Optional path to the matching ``_vecnormalize.pkl`` file.  When
        present, observations are normalised with the same running statistics
        that were used during training.
    """

    def __init__(
        self,
        act_ctrl: "ActionController",
        ckpt_path: Path,
        vecnorm_path: Path | None = None,
    ) -> None:
        super().__init__(act_ctrl)
        self.ckpt_path = ckpt_path
        logger.info("Loading RL policy from %s (device=cpu) …", ckpt_path)

        # PyTorch distributions validate_args can cause Simplex constraint failures
        # due to float32 precision errors during MaskableCategorical softmax normalization
        torch.distributions.Distribution.set_default_validate_args(False)

        # custom_objects keeps the shapes/hypers stable even if the saved
        # config differs from the current training_config.yaml.
        self._model: MaskablePPO = MaskablePPO.load(  # type: ignore[assignment]
            ckpt_path,
            device="cpu",
            custom_objects={},
        )

        self._vec_normalize: VecNormalize | None = None
        if vecnorm_path is not None and vecnorm_path.exists():
            logger.info("Loading VecNormalize stats from %s …", vecnorm_path)
            # We only need the normalisation statistics, not a real env.
            # Load with a dummy DummyVecEnv wrapper is avoided; instead we
            # just load the pkl and apply obs normalisation manually.
            with open(vecnorm_path, "rb") as f:
                self._vec_normalize = pickle.load(f)
            self._vec_normalize.training = False  # inference mode — no stat updates
        elif vecnorm_path is not None:
            logger.warning(
                "VecNormalize file %s not found — observations will not be normalised.",
                vecnorm_path,
            )

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self, state_data: m.EnvironmentStateData) -> npt.NDArray[np.float32]:
        """Build the flat observation vector from *state_data*.

        Mirrors :meth:`src.share.env.BaseMIGResourceEnv._get_obs` exactly so
        the policy sees the same feature layout it was trained on.
        """
        obs_list: List[float] = []
        agents_ordered = sorted(list(m.AgentId), key=lambda a: a.value)

        scalar_metrics = [
            "arrival_rate",
            "predicted_arrival_rate",
            "total_sm_ratio",
            "total_vram_ratio",
        ]

        for aid in agents_ordered:
            # 4 scalar metrics
            for metric in scalar_metrics:
                data = cast(Dict[m.AgentId, float], state_data[metric])  # type: ignore
                obs_list.append(float(data[aid]))

            # arrival_rate_history (history_len values)
            obs_list.extend(state_data["arrival_rate_history"][aid])

            # 7 KV cache utilisation
            obs_list.extend(state_data["kv_cache_utilization"][aid])

            # 7 avg composite latency proportions
            obs_list.extend(state_data["avg_composite_latency"][aid])

            # 7 avg queue length (log-normalised)
            obs_list.extend(state_data["avg_queue_length"][aid])

            # 7 avg queue length trend
            obs_list.extend(state_data["avg_queue_length_trend"][aid])

            # 7 avg running requests
            obs_list.extend(state_data["avg_running_requests"][aid])

            # 6 agent-owns-MIG per-profile counts
            obs_list.extend([float(x) for x in state_data["agent_owns_mig"][aid]])

            # 6 action history metrics
            obs_list.append(float(state_data["last_split"][aid]))
            obs_list.append(float(state_data["last_merge"][aid]))
            obs_list.append(float(state_data["last_give"][aid]))
            obs_list.append(float(state_data["last_receive"][aid]))
            obs_list.append(float(state_data["last_give_amount"][aid]))
            obs_list.append(float(state_data["last_receive_amount"][aid]))

        # Global metrics
        obs_list.append(1.0 if state_data["recovery_flag"] else 0.0)
        obs_list.append(float(state_data["current_budget"]))
        obs_list.append(float(state_data["downtime_ratio"]))

        # Agent ratios (CODING − RAG)
        obs_list.append(float(state_data["agent_arrival_rate_ratio"]))
        obs_list.append(float(state_data["agent_avg_queue_len_ratio"]))
        obs_list.append(float(state_data["agent_avg_running_req_ratio"]))
        obs_list.append(float(state_data["agent_avg_kv_cache_ratio"]))
        obs_list.append(float(state_data["agent_avg_composite_latency_ratio"]))
        obs_list.append(0.0)  # placeholder kept for training parity
        obs_list.append(float(state_data["agent_vram_ratio"]))
        obs_list.append(float(state_data["agent_sm_ratio"]))

        # MIG geometry: GPU 0 [coding, rag], GPU 1 [coding, rag]
        mig_geom = state_data["mig_geometry"]
        for gpu_idx in (0, 1):
            for s in mig_geom.get(gpu_idx, [0.0, 0.0]):
                obs_list.append(float(s))

        # MIG profile one-hot: 15 values per GPU
        onehot = state_data["mig_profile_id_onehot"]
        for gpu_idx in (0, 1):
            obs_list.extend(onehot.get(gpu_idx, [0.0] * 15))

        # Ownership grid: 7 values per GPU
        grid = state_data["ownership_grid"]
        for gpu_idx in (0, 1):
            obs_list.extend([float(x) for x in grid.get(gpu_idx, [0] * 7)])

        obs = np.array(obs_list, dtype=np.float32)

        # Apply VecNormalize statistics if available
        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        return obs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(
        self,
        state_data: m.EnvironmentStateData,
        action_mask: List[bool],
    ) -> m.ResourceManagerAction:
        """Return the policy's chosen :class:`~src.share.models.ResourceManagerAction`.

        Parameters
        ----------
        state_data:
            Current environment state from :meth:`~src.deploy.obs.ObservationCollector.get_observation`.
        action_mask:
            Boolean mask produced by :meth:`~src.deploy.act_controller.ActionController.get_action_mask`.
        """
        obs = self._get_obs(state_data)
        mask_arr = np.array(action_mask, dtype=np.bool_)

        # MaskablePPO.predict expects a batched obs; squeeze the extra dim back.
        action_idx, _ = self._model.predict(
            obs,
            action_masks=mask_arr,
            deterministic=True,
        )
        # predict() returns an ndarray when given a single obs
        idx = int(action_idx)
        chosen = list(m.ResourceManagerAction)[idx]
        logger.debug("RL policy chose action %d → %s", idx, chosen.name)
        return chosen

    # ------------------------------------------------------------------
    # Deployment control loop
    # ------------------------------------------------------------------

    async def run_loop(
        self,
        duration_s: float,
    ) -> None:
        """Periodically collect an observation and execute the policy's action.

        Fires every ``TRAINING_CONFIG.action_interval`` seconds, mirroring the
        ``RESOURCE_MANAGER_TRIGGER`` cadence in the simulation.  Must be
        awaited concurrently with the request-dispatch future returned by
        :meth:`~src.deploy.req_pub.ReqPublisher.start_sending`.

        Parameters
        ----------
        duration_s:
            Total benchmark duration in seconds; the loop exits when this
            wall-clock time has elapsed.
        """
        OBS_COLLECTOR.start_budget_refresh_loop()
        start = time.time()
        step = 0

        logger.info(
            "RL control loop started (duration=%.0fs, interval=%.0fs).",
            duration_s,
            TRAINING_CONFIG.action_interval,
        )

        while True:
            # Sleep until the next action trigger point
            next_trigger = start + (step + 1) * TRAINING_CONFIG.action_interval
            sleep_for = next_trigger - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

            if time.time() - start >= duration_s:
                logger.info("RL control loop finished after %d steps.", step)
                break

            # Finalise the interval's accumulated metrics
            OBS_COLLECTOR.start_new_interval()

            # Observation and action mask
            state_data = OBS_COLLECTOR.get_observation()
            action_mask = self.act_ctrl.get_action_mask()

            # Log the complete action mask
            allowed_action_names = [
                act.name
                for i, act in enumerate(m.ResourceManagerAction)
                if action_mask[i]
            ]
            logger.info(
                "[step %d] Complete Action Mask (1s and 0s): %s",
                step,
                "".join("1" if x else "0" for x in action_mask),
            )
            logger.info(
                "[step %d] Allowed actions (%d/%d): %s",
                step,
                len(allowed_action_names),
                len(action_mask),
                allowed_action_names,
            )

            # Policy inference
            chosen_action = self._predict(state_data, action_mask)
            logger.info("[step %d] RL action: %s", step, chosen_action.name)

            # Execute (NO_ACTION is a no-op)
            if chosen_action != m.ResourceManagerAction.NO_ACTION:
                concrete_action = self.act_ctrl.map_to_action(chosen_action)
                if concrete_action is not None:
                    await self.act_ctrl.execute_action(concrete_action)

            step += 1
