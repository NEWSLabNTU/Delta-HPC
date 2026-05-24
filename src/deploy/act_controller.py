import time
import logging
import asyncio
import subprocess
from typing import List, Optional

import src.share.models as m
from src.share.mig_matrix import (
    STATE_DEFINITIONS,
    TRANSITION_MATRIX,
    SLICE_MAPPING,
    map_res_action_to_action,
)
from src.deploy.system import SYSTEM_STATE, register_gpu
from src.deploy.vllm import VLLMManager
from src.deploy.mig_controller import MIGController
from src.deploy.obs import OBS_COLLECTOR
from src.deploy.metrics import VLLMMetricsClient
from src.deploy.models import MIGSlotState, ProfilePlacement
from src.training.config import TRAINING_CONFIG
from src.simulation.utils import SIM_CONFIG
from src.simulation.config import (
    GPU_MIG_PROFILE,
    GPU_VALID_COMBINATIONS,
)

logger = logging.getLogger(__name__)


class ActionController:
    """Translates and executes RL actions on physical hardware.

    This class bridges the high-level reinforcement learning actions with the
    low-level MIG and vLLM management APIs. It also handles action masking
    and budget tracking to maintain parity with the simulation environment.
    """

    def __init__(self, vllm_mgr: VLLMManager):
        self.vllm_mgr = vllm_mgr
        gpu_indices = list(SYSTEM_STATE.gpus.keys())
        self.mig_ctrl = MIGController(gpu_indices=gpu_indices)

    def get_action_mask(self, ignore_cooldowns: bool = False) -> List[bool]:
        """
        Evaluate and return a boolean mask for all m.ResourceManagerAction enum values.

        Logic follows src/simulation/simulator.py:909.
        """
        mask = [False] * len(m.ResourceManagerAction)

        # 0. Global Constraints
        if OBS_COLLECTOR.reconfig_flag:
            mask[
                list(m.ResourceManagerAction).index(m.ResourceManagerAction.NO_ACTION)
            ] = True
            return mask

        cooldown_steps = TRAINING_CONFIG.action_cooldown if not ignore_cooldowns else 0
        transfer_blocked = any(
            OBS_COLLECTOR._agent_stats[aid].action_history["give"]["intervals"]
            < cooldown_steps
            for aid in list(m.AgentId)
        )

        current_states = {
            gpu_idx: self._identify_gpu_state(gpu_idx)
            for gpu_idx, gpu_state in SYSTEM_STATE.gpus.items()
            if not gpu_state.is_simulated
        }

        for act_id, action in enumerate(m.ResourceManagerAction):
            if action == m.ResourceManagerAction.NO_ACTION:
                mask[act_id] = True
                continue

            val = action.value
            act_gpu_id, target_sid, trans_mig = (
                val.gpu_id,
                val.target_state_id,
                val.transfer_mig,
            )

            if act_gpu_id not in current_states:
                continue

            current_sid = current_states[act_gpu_id]

            # 1. Hardware & Transition Validity
            if target_sid is not None:
                # Check if target state is valid for this specific GPU model
                if (
                    STATE_DEFINITIONS[target_sid]
                    not in GPU_VALID_COMBINATIONS[act_gpu_id]
                ):
                    continue

                # GLOBAL CONSTRAINT: Avoid states with unsupported profiles (not supported by current RL model for this GPU)
                unsupported = GPU_MIG_PROFILE[act_gpu_id].unsupported_profiles()
                if any(p in unsupported for p in STATE_DEFINITIONS[target_sid]):
                    continue

                # Check if transition exists in matrix
                trans_action = TRANSITION_MATRIX.get((current_sid, target_sid))
                if not trans_action:
                    continue

                # RECONFIGURATION CONSTRAINT: All source engines must have the same owner
                gpu_state = SYSTEM_STATE.gpus[act_gpu_id]
                src_indices = trans_action.mig_src
                owners = {
                    gpu_state.slots[idx].agent_id
                    for idx in src_indices
                    if idx < len(gpu_state.slots)
                }
                if len(owners) > 1:
                    continue

            # 2. Transfer Validity
            if trans_mig is not None:
                # Check hardware support
                supported_profiles = {
                    p.profile_type for p in GPU_MIG_PROFILE[act_gpu_id]
                }
                unsupported = GPU_MIG_PROFILE[act_gpu_id].unsupported_profiles()
                if trans_mig not in supported_profiles or trans_mig in unsupported:
                    continue

                if transfer_blocked:
                    continue

                if target_sid is None:
                    # Pure transfer: Check if GPU has the requested MIG profile (not permanent)
                    has_mig = any(
                        slot.profile_placement.profile.profile_type == trans_mig
                        for slot in SYSTEM_STATE.gpus[act_gpu_id].slots
                    )
                    mask[act_id] = has_mig
                else:
                    # State + transition: Transfer MIG must be one of the resulting profiles
                    target_profiles = STATE_DEFINITIONS[target_sid]
                    resulting_profiles = [
                        target_profiles[idx]
                        for idx in TRANSITION_MATRIX[
                            (current_sid, target_sid)
                        ].mig_target
                    ]
                    if trans_mig in resulting_profiles:
                        mask[act_id] = True
            else:
                # Pure state transition
                mask[act_id] = True

            # 3. Budget Check & Slot Readiness
            if mask[act_id]:
                pred_action = self.map_to_action(action)
                if pred_action:
                    # Disable action if any involved slot is restarting / not ready
                    gpu_state = SYSTEM_STATE.gpus[pred_action.gpu_id]
                    if any(
                        not gpu_state.slots[idx].is_ready for idx in pred_action.mig_src
                    ):
                        mask[act_id] = False
                        continue

                    cost = self.predict_action_cost(pred_action)
                    if cost > OBS_COLLECTOR.current_budget:
                        mask[act_id] = False

        return mask

    def map_to_action(self, res_action: m.ResourceManagerAction) -> Optional[m.Action]:
        """Convert a ResourceManagerAction enum to a concrete Action object."""
        gpu_id = res_action.value.gpu_id
        current_sid = self._identify_gpu_state(gpu_id)

        return map_res_action_to_action(
            res_action=res_action,
            current_sid=current_sid,
            find_best_index_fn=self._find_best_slot_index,
            get_owner_fn=self._get_slot_owner,
        )

    def predict_action_cost(self, action: m.Action) -> float:
        """Estimate the downtime cost (drain + boot) of an action."""
        # We reuse simulation cost constants for policy consistency

        cost = 0.0
        if action.action == m.ActionType.TRANSFER:
            slot = SYSTEM_STATE.gpus[action.gpu_id].slots[action.mig_src[0]]
            drain_cost = self._predict_drain_time(slot)
            boot_cost = SIM_CONFIG.get_restart_time(
                action.receiver.receiver_id,
                slot.profile_placement.profile,
                gpu_id=action.gpu_id,
            )
            cost = drain_cost + boot_cost
        elif action.action in [m.ActionType.SPLIT, m.ActionType.MERGE]:
            gpu_state = SYSTEM_STATE.gpus[action.gpu_id]
            affected_slots = [gpu_state.slots[idx] for idx in action.mig_src]
            drain_cost = max(
                [self._predict_drain_time(s) for s in affected_slots], default=0.0
            )

            boot_costs = []
            target_sid = action.target_state_id
            for target_idx in action.mig_target:
                profile_type = STATE_DEFINITIONS[target_sid][target_idx]
                hw_profile = self._get_hardware_profile(action.gpu_id, profile_type)

                # Owner defaults to current owner unless it's the receiver's slot
                owner_id = affected_slots[0].agent_id
                if action.receiver and target_idx == action.receiver.mig_idx:
                    owner_id = action.receiver.receiver_id

                boot_costs.append(
                    SIM_CONFIG.get_restart_time(
                        owner_id, hw_profile, gpu_id=action.gpu_id
                    )
                )

            boot_cost = max(boot_costs, default=0.0)
            cost = drain_cost + boot_cost

        return cost

    async def execute_action(self, action: m.Action):
        """Execute the action on physical hardware and update system state."""
        gpu_id = action.gpu_id
        gpu_state = SYSTEM_STATE.gpus[gpu_id]
        assert not gpu_state.is_simulated, (
            f"GPU {gpu_id} is simulated and cannot be reconfigured."
        )

        # 1. Update Telemetry & Budget
        cost = self.predict_action_cost(action)
        OBS_COLLECTOR.consume_budget(cost)

        # Record to reconfiguration history
        src_profiles = [
            gpu_state.slots[idx].profile_placement.profile.profile_type.name
            for idx in action.mig_src
        ]
        target_profiles = [
            STATE_DEFINITIONS[action.target_state_id][idx].name
            for idx in action.mig_target
        ]
        details = (
            f"GPU {gpu_id} | Src: {', '.join(src_profiles)} "
            f"-> Tgt: {', '.join(target_profiles)}"
        )
        if action.receiver:
            details += f" | Receiver: {action.receiver.receiver_id.name}"

        OBS_COLLECTOR.record_reconfig(action.action.name, cost, details)

        src_indices = action.mig_src
        triggering_agent = gpu_state.slots[src_indices[0]].agent_id

        # Increment split/merge/transfer counts
        if action.action == m.ActionType.SPLIT:
            OBS_COLLECTOR.increment_reconfig_count(triggering_agent, "split")
        elif action.action == m.ActionType.MERGE:
            OBS_COLLECTOR.increment_reconfig_count(triggering_agent, "merge")
        if action.receiver is not None:
            OBS_COLLECTOR.increment_reconfig_count(triggering_agent, "transfer")

        if action.action == m.ActionType.TRANSFER:
            vram = gpu_state.slots[src_indices[0]].profile_placement.profile.vram
            OBS_COLLECTOR.set_last_action(triggering_agent, "give", vram)
            OBS_COLLECTOR.set_last_action(action.receiver.receiver_id, "receive", vram)
        else:
            OBS_COLLECTOR.set_last_action(triggering_agent, action.action.name.lower())
            if action.receiver:
                target_profile = STATE_DEFINITIONS[action.target_state_id][
                    action.receiver.mig_idx
                ]
                hw_prof = self._get_hardware_profile(gpu_id, target_profile)
                OBS_COLLECTOR.set_last_action(triggering_agent, "give", hw_prof.vram)
                OBS_COLLECTOR.set_last_action(
                    action.receiver.receiver_id, "receive", hw_prof.vram
                )

        # 2. Teardown affected slots only
        stop_tasks = []
        for idx in src_indices:
            slot = gpu_state.slots[idx]
            slot.is_draining = True
            logger.info(
                f"Draining and stopping vLLM container: GPU {gpu_id} start_slice={slot.profile_placement.start_slice}"
            )
            stop_tasks.append(asyncio.to_thread(self.vllm_mgr.stop, slot))

        if stop_tasks:
            await asyncio.gather(*stop_tasks)
            time.sleep(2.0)

        # 3. MIG Reconfiguration (Partial)
        fallback_triggered = False
        original_owners = {}
        if action.action != m.ActionType.TRANSFER:
            # Capture original ownership of all slots on this physical GPU in case fallback is triggered
            original_owners = {
                s.profile_placement.start_slice: s.agent_id for s in gpu_state.slots
            }

            # Destroy affected instances
            start_slices = [
                gpu_state.slots[i].profile_placement.start_slice for i in src_indices
            ]
            try:
                await asyncio.to_thread(
                    self.mig_ctrl.destroy_gis_at_slices, gpu_id, start_slices
                )
            except Exception as e:
                # Check for exit code 19 / "In use by another client"
                is_in_use_error = False
                if isinstance(e, subprocess.CalledProcessError):
                    err_msg = e.stderr.decode().strip() if e.stderr else ""
                    if e.returncode == 19 or "In use by another client" in err_msg:
                        is_in_use_error = True

                if not is_in_use_error:
                    raise e

                logger.warning(
                    f"GPU {gpu_id}: MIG reconfiguration failed due to active CUDA context ({e}). "
                    "Initiating fallback: gracefully stopping ALL other containers on this physical GPU..."
                )
                fallback_triggered = True

                # Step 1: Drain and stop all other active containers on this physical GPU
                fallback_stop_tasks = []
                for slot in gpu_state.slots:
                    # If it has a mig_uuid and is not already draining/stopped, stop it
                    if slot.mig_uuid and not slot.is_draining:
                        slot.is_draining = True
                        logger.info(
                            f"Fallback: stopping preserved container on GPU {gpu_id} start_slice={slot.profile_placement.start_slice}"
                        )
                        fallback_stop_tasks.append(
                            asyncio.to_thread(self.vllm_mgr.stop, slot)
                        )

                if fallback_stop_tasks:
                    await asyncio.gather(*fallback_stop_tasks)

                # Step 2: Retry destroy_gis_at_slices (which should now succeed perfectly!)
                logger.info(
                    f"GPU {gpu_id}: retrying MIG reconfiguration after clearing all physical GPU contexts..."
                )
                await asyncio.to_thread(
                    self.mig_ctrl.destroy_gis_at_slices, gpu_id, start_slices
                )

            # Create new instances
            target_sid = action.target_state_id
            target_profiles = STATE_DEFINITIONS[target_sid]
            target_slices = SLICE_MAPPING[target_sid]

            for target_idx in action.mig_target:
                profile_type = target_profiles[target_idx]
                start_slice = target_slices[target_idx][0]
                hw_prof = self._get_hardware_profile(gpu_id, profile_type)
                logger.info(
                    f"GPU {gpu_id}: creating instance profile={profile_type.name} start_slice={start_slice}"
                )
                await asyncio.to_thread(
                    self.mig_ctrl.create_gi, gpu_id, hw_prof, start_slice
                )

            # 4. State Reconstruction
            uuids = await asyncio.to_thread(self.mig_ctrl.list_mig_device_uuids, gpu_id)
        # 4. State Update and Booting
        match action.action:
            case m.ActionType.TRANSFER:
                idx = action.mig_src[0]
                slot = gpu_state.slots[idx]

                # Update ownership
                slot.agent_id = action.receiver.receiver_id
                register_gpu(gpu_state)

                logger.info(
                    f"Starting vLLM: GPU {gpu_id} slice={slot.profile_placement.start_slice} agent={slot.agent_id}"
                )
                await asyncio.to_thread(self.vllm_mgr.start, slot)
                await asyncio.to_thread(self.vllm_mgr.wait_until_ready, slot)

            case m.ActionType.SPLIT | m.ActionType.MERGE:
                # Remove old destroyed slots (reverse order to preserve indices during pop)
                for idx in sorted(src_indices, reverse=True):
                    gpu_state.slots.pop(idx)

                target_sid = action.target_state_id
                target_profiles = STATE_DEFINITIONS[target_sid]
                target_slices = SLICE_MAPPING[target_sid]

                new_slots_to_start = []
                for target_idx in action.mig_target:
                    profile_type = target_profiles[target_idx]
                    start_slice = target_slices[target_idx][0]
                    uuid = next((u for s, u in uuids if s == start_slice), None)
                    hw_prof = self._get_hardware_profile(gpu_id, profile_type)

                    owner = triggering_agent
                    if action.receiver and target_idx == action.receiver.mig_idx:
                        owner = action.receiver.receiver_id

                    new_slot = MIGSlotState(
                        gpu_idx=gpu_id,
                        profile_placement=ProfilePlacement(hw_prof, start_slice),
                        mig_uuid=uuid,
                        agent_id=owner,
                        is_ready=False,
                    )
                    gpu_state.slots.append(new_slot)
                    new_slots_to_start.append(new_slot)

                # Restore original ownership and queue preserved slots for boot if fallback was triggered
                if fallback_triggered:
                    for slot in gpu_state.slots:
                        if slot not in new_slots_to_start:
                            orig_owner = original_owners.get(
                                slot.profile_placement.start_slice
                            )
                            if orig_owner is not None:
                                slot.agent_id = orig_owner
                            new_slots_to_start.append(slot)

                # Re-sort to maintain physical layout order
                gpu_state.slots.sort(key=lambda s: s.profile_placement.start_slice)
                register_gpu(gpu_state)

                # Boot new containers concurrently
                start_tasks = []
                for slot in new_slots_to_start:
                    logger.info(
                        f"Starting vLLM: GPU {gpu_id} slice={slot.profile_placement.start_slice} agent={slot.agent_id}"
                    )
                    start_tasks.append(asyncio.to_thread(self.vllm_mgr.start, slot))
                if start_tasks:
                    await asyncio.gather(*start_tasks)

                # Wait for readiness concurrently
                ready_tasks = []
                for slot in new_slots_to_start:
                    ready_tasks.append(
                        asyncio.to_thread(self.vllm_mgr.wait_until_ready, slot)
                    )
                if ready_tasks:
                    await asyncio.gather(*ready_tasks)

        OBS_COLLECTOR.mark_reconfig_complete()
        logger.info(f"Action {action.action.name} complete on GPU {gpu_id}")

    def _identify_gpu_state(self, gpu_id: int) -> int:
        gpu_state = SYSTEM_STATE.gpus.get(gpu_id)
        if not gpu_state:
            raise ValueError(f"GPU {gpu_id} not registered")

        profiles = sorted(
            [s.profile_placement.profile.profile_type for s in gpu_state.slots],
            key=lambda x: x.value,
        )
        for sid, defs in STATE_DEFINITIONS.items():
            if sorted(list(defs), key=lambda x: x.value) == profiles:
                return sid
        raise ValueError(f"Unknown MIG state for GPU {gpu_id}")

    def _get_slot_owner(self, gpu_id: int, slot_idx: int) -> m.AgentId:
        return SYSTEM_STATE.gpus[gpu_id].slots[slot_idx].agent_id

    def _find_best_slot_index(self, gpu_id: int, profile_type: m.MIGProfile) -> int:
        candidates = []
        for i, slot in enumerate(SYSTEM_STATE.gpus[gpu_id].slots):
            if slot.profile_placement.profile.profile_type == profile_type:
                q_len = OBS_COLLECTOR.get_last_queue_length(
                    gpu_id, slot.profile_placement.start_slice
                )
                candidates.append((i, q_len))

        if not candidates:
            return -1

        # Pick the least busy slot (matches simulation heuristic)
        idx, _ = min(candidates, key=lambda x: x[1])
        return idx

    def _get_hardware_profile(
        self, gpu_id: int, profile_type: m.MIGProfile
    ) -> m.MIGProfileBase:
        for hp in GPU_MIG_PROFILE[gpu_id]:
            if hp.profile_type == profile_type:
                return hp
        raise ValueError(f"No hardware profile for {profile_type} on GPU {gpu_id}")

    def _predict_drain_time(self, slot: MIGSlotState) -> float:
        """Estimate the time to drain in-flight requests from a slot."""

        if slot.port is None:
            return 0.0

        try:
            # 1. Get real-time per-request token progress from VLLMManager
            running_tokens = self.vllm_mgr.get_running_requests_tokens(slot.mig_uuid)

            client = VLLMMetricsClient(slot.port, timeout=1.0)
            data = client.collect()
            running_count = data.get("running_requests", 0)

            if running_count == 0:
                return 0.0

            # Estimation: (avg_response_len - current_progress) * tpot
            avg_len = OBS_COLLECTOR.get_avg_response_len(slot.agent_id)
            tpot = OBS_COLLECTOR.get_current_tpot(slot.agent_id)

            if running_tokens:
                # The request with the fewest tokens is furthest from completion.
                # We assume its total length will be avg_len.
                min_tokens = min(running_tokens)
                remaining_tokens = max(0.0, avg_len - min_tokens)
            else:
                # Fallback if tracking is empty but metrics show running requests
                # (e.g. requests sent from outside this ActionController's VLLMManager)
                remaining_tokens = avg_len

            return remaining_tokens * tpot
        except Exception as e:
            logger.debug(
                f"Could not fetch metrics for drain prediction on port {slot.port}: {e}"
            )
            return 5.0  # Fallback guess
