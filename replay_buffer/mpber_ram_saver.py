import ray
import zlib
import logging
import numpy as np
from itertools import chain
from gymnasium.spaces import Space
from typing import Dict, Optional
from ray.rllib.utils.typing import PolicyID
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.replay_buffers.utils import SampleBatchType
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode, merge_dicts_with_warning
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import _ALL_POLICIES
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from replay_buffer.replay_node import BaseBuffer
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, concat_samples
from ray.util.debug import log_once
from utils import split_list_into_n_parts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decompress_data(data, length, batch_size):
    compressed_item = data[:length]
    decompressed_bytes = zlib.decompress(compressed_item)
    return np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape([batch_size, -1])


def decompress_sample_block(ma_block):
    n = ma_block["terminateds"].shape[0]
    obs_bytes = ma_block["obs"].reshape(-1)
    new_obs_bytes = ma_block["new_obs"].reshape(-1)
    obs = decompress_data(obs_bytes, ma_block["length_obs"][0], n)
    new_obs = decompress_data(new_obs_bytes, ma_block["length_new_obs"][0], n)
    data = SampleBatch(
        {
            "obs": obs,
            "new_obs": new_obs,
            "actions": ma_block["actions"],
            "rewards": ma_block["rewards"],
            "terminateds": ma_block["terminateds"],
            "truncateds": ma_block["truncateds"],
            "weights": ma_block["weights"],
            "batch_indexes": ma_block["batch_indexes"],
        }
    )
    return data


@ray.remote(num_cpus=2, max_calls=50, num_returns=1)
def compress_sample_block_loop(samples, store):
    result = []
    for each in samples:
        result.append(compress_sample_block(each[0], each[1], store))
    return result


def compress_sample_block(sample_block, weight, store):
    obs = np.array(sample_block["obs"])
    new_obs = np.array(sample_block["new_obs"])
    n = obs.shape[0]

    obs_bytes = obs.tobytes()
    new_obs_bytes = new_obs.tobytes()

    compressed_obs = zlib.compress(obs_bytes, 8)
    compressed_new_obs = zlib.compress(new_obs_bytes, 8)
    length_obs = len(compressed_obs)
    length_new_obs = len(compressed_new_obs)

    total_capacity = n * store
    if length_obs > total_capacity:
        raise ValueError(f"Compressed obs size {length_obs} exceeds N*store ({n}*{store})")
    if length_new_obs > total_capacity:
        raise ValueError(f"Compressed new_obs size {length_new_obs} exceeds N*store ({n}*{store})")

    padded_obs = np.zeros((n * store,), dtype=np.uint8)
    padded_obs[:length_obs] = np.frombuffer(compressed_obs, dtype=np.uint8)
    padded_obs = padded_obs.reshape(n, store)

    padded_new_obs = np.zeros((n * store,), dtype=np.uint8)
    padded_new_obs[:length_new_obs] = np.frombuffer(compressed_new_obs, dtype=np.uint8)
    padded_new_obs = padded_new_obs.reshape(n, store)

    data = SampleBatch(
        {
            "obs": padded_obs,
            "new_obs": padded_new_obs,
            "actions": sample_block["actions"],
            "rewards": sample_block["rewards"],
            "terminateds": sample_block["terminateds"],
            "truncateds": sample_block["truncateds"],
            "weights": sample_block["weights"],
            "length_obs": np.array([length_obs], dtype=np.int32),
            "length_new_obs": np.array([length_new_obs], dtype=np.int32),
        }
    )
    return data, weight


class PrioritizedBlockReplayBuffer(PrioritizedReplayBuffer):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            randomly: bool = False,
            sub_buffer_size: int = 32,
            beta=0.6,
            store=2000,
            num_save=200,
            split_mini_batch=10,
            **kwargs
    ):
        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)
        self.beta = beta
        self.store = store
        self.num_save = num_save
        self.split_mini_batch = split_mini_batch
        self.base_buffer = BaseBuffer(sub_buffer_size, obs_space, action_space, randomly)
        self._sub_store = []

    def sample(self, num_items: int, **kwargs):
        return super(PrioritizedBlockReplayBuffer, self).sample(num_items, **kwargs)

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch of experiences to this buffer.

        Splits batch into chunks of timesteps, sequences or episodes, depending on
        `self._storage_unit`. Calls `self._add_single_batch` to add resulting slices
        to the buffer storage.

        Args:
            batch: Batch to add.
            ``**kwargs``: Forward compatibility kwargs.
        """
        if not batch.count > 0:
            return
        buffer = self.base_buffer

        self.base_buffer.add(batch)
        if buffer.full:
            data = buffer.sample()
            weight = np.mean(data.get("weights"))
            buffer.reset()
            self._sub_store.append([data, weight])
        if len(self._sub_store) == self.num_save:
            _list = split_list_into_n_parts(self._sub_store, n=self.split_mini_batch)
            result_ids = [compress_sample_block_loop.remote(batch, self.store) for batch in _list]
            results = ray.get(result_ids)
            results = list(chain(*results))
            for each in results:
                self._add_single_batch(each[0], weight=each[1])

            self._sub_store = []


@DeveloperAPI
class MultiAgentPrioritizedBlockReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """A prioritized replay buffer shard for multiagent setups.

    This buffer is meant to be run in parallel to distribute experiences
    across `num_shards` shards. Unlike simpler buffers, it holds a set of
    buffers - one for each policy ID.
    """

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 1,
            rollout_fragment_length: int = 4,
            capacity: int = 10000,
            storage_unit: str = "timesteps",
            num_shards: int = 1,
            replay_mode: str = "independent",
            replay_sequence_override: bool = True,
            replay_sequence_length: int = 1,
            replay_burn_in: int = 0,
            replay_zero_init_states: bool = True,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_beta: float = 0.4,
            prioritized_replay_eps: float = 1e-6,
            split_mini_batch=10,
            num_save=200,
            store=2000,
            **kwargs
    ):
        """Initializes a MultiAgentReplayBuffer instance.

        Args:
            capacity: The capacity of the buffer, measured in `storage_unit`.
            storage_unit: Either 'timesteps', 'sequences' or
                'episodes'. Specifies how experiences are stored. If they
                are stored in episodes, replay_sequence_length is ignored.
                If they are stored in episodes, replay_sequence_length is
                ignored.
            num_shards: The number of buffer shards that exist in total
                (including this one).
            learning_starts: Number of timesteps after which a call to
                `replay()` will yield samples (before that, `replay()` will
                return None).
            replay_mode: One of "independent" or "lockstep". Determines,
                whether batches are sampled independently or to an equal
                amount.
            replay_sequence_override: If True, ignore sequences found in incoming
                batches, slicing them into sequences as specified by
                `replay_sequence_length` and `replay_sequence_burn_in`. This only has
                an effect if storage_unit is `sequences`.
            replay_sequence_length: The sequence length (T) of a single
                sample. If > 1, we will sample B x T from this buffer.
            replay_burn_in: The burn-in length in case
                `replay_sequence_length` > 0. This is the number of timesteps
                each sequence overlaps with the previous one to generate a
                better internal state (=state after the burn-in), instead of
                starting from 0.0 each RNN rollout.
            replay_zero_init_states: Whether the initial states in the
                buffer (if replay_sequence_length > 0) are always 0.0 or
                should be updated with the previous train_batch state outputs.
            underlying_buffer_config: A config that contains all necessary
                constructor arguments and arguments for methods to call on
                the underlying buffers. This replaces the standard behaviour
                of the underlying PrioritizedReplayBuffer. The config
                follows the conventions of the general
                replay_buffer_config. kwargs for subsequent calls of methods
                may also be included. Example:
                "replay_buffer_config": {"type": PrioritizedReplayBuffer,
                "capacity": 10, "storage_unit": "timesteps",
                prioritized_replay_alpha: 0.5, prioritized_replay_beta: 0.5,
                prioritized_replay_eps: 0.5}
            prioritized_replay_alpha: Alpha parameter for a prioritized
                replay buffer. Use 0.0 for no prioritization.
            prioritized_replay_beta: Beta parameter for a prioritized
                replay buffer.
            prioritized_replay_eps: Epsilon parameter for a prioritized
                replay buffer.
            ``**kwargs``: Forward compatibility kwargs.
        """

        if "replay_mode" in kwargs and (
                kwargs["replay_mode"] == "lockstep"
                or kwargs["replay_mode"] == ReplayMode.LOCKSTEP
        ):
            if log_once("lockstep_mode_not_supported"):
                logger.error(
                    "Replay mode `lockstep` is not supported for "
                    "MultiAgentPrioritizedReplayBuffer. "
                    "This buffer will run in `independent` mode."
                )
            kwargs["replay_mode"] = "independent"
        pber_config = {
            "type": PrioritizedBlockReplayBuffer,
            "action_space": action_space,
            "obs_space": obs_space,
            "storage_unit": StorageUnit.FRAGMENTS,
            "randomly": False,
            "sub_buffer_size": sub_buffer_size,
            "alpha": prioritized_replay_alpha,
            "beta": prioritized_replay_beta,
            "split_mini_batch": split_mini_batch,
            "num_save": num_save,
            "store": store,
        }
        MultiAgentPrioritizedReplayBuffer.__init__(
            self,
            capacity=capacity,
            storage_unit=storage_unit,
            num_shards=num_shards,
            replay_mode=replay_mode,
            replay_sequence_override=replay_sequence_override,
            replay_sequence_length=replay_sequence_length,
            replay_burn_in=replay_burn_in,
            replay_zero_init_states=replay_zero_init_states,
            underlying_buffer_config=pber_config,
            prioritized_replay_alpha=prioritized_replay_alpha,
            prioritized_replay_beta=prioritized_replay_beta,
            prioritized_replay_eps=prioritized_replay_eps,
            **kwargs,
        )
        self.rollout_fragment_length = rollout_fragment_length

    @DeveloperAPI
    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch to the appropriate policy's replay buffer.

        Turns the batch into a MultiAgentBatch of the DEFAULT_POLICY_ID if
        it is not a MultiAgentBatch. Subsequently, adds the individual policy
        batches to the storage.

        Args:
            batch : The batch to be added.
            ``**kwargs``: Forward compatibility kwargs.
        """
        if batch is None:
            if log_once("empty_batch_added_to_buffer"):
                logger.info(
                    "A batch that is `None` was added to {}. This can be "
                    "normal at the beginning of execution but might "
                    "indicate an issue.".format(type(self).__name__)
                )
            return
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multi-agent.
        batch = batch.as_multi_agent()
        with self.add_batch_timer:
            pids_and_batches = self._maybe_split_into_policy_batches(batch)
            for policy_id, sample_batch in pids_and_batches.items():
                if len(sample_batch) == 1:
                    self._add_to_underlying_buffer(policy_id, sample_batch)
                else:
                    _ = sample_batch.timeslices(size=self.rollout_fragment_length)
                    for s_batch in _:
                        self._add_to_underlying_buffer(policy_id, s_batch)

        self._num_added += batch.count

    @DeveloperAPI
    @override(PrioritizedReplayBuffer)
    def update_priorities(self, prio_dict: Dict) -> None:
        """Updates the priorities of underlying replay buffers.

        Computes new priorities from td_errors and prioritized_replay_eps.
        These priorities are used to update underlying replay buffers per
        policy_id.

        Args:
            prio_dict: A dictionary containing td_errors for
            batches saved in underlying replay buffers.
        """
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                self.replay_buffers[policy_id].update_priorities(
                    batch_indexes, new_priorities
                )

    @DeveloperAPI
    @override(ReplayBuffer)
    def sample(
            self, num_items: int, policy_id: Optional[PolicyID] = None, **kwargs
    ) -> Optional[SampleBatchType]:
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        with self.replay_timer:
            # Lockstep mode: Sample from all policies at the same time an
            # equal amount of steps.
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert (
                        policy_id is None
                ), "`policy_id` specifier not allowed in `lockstep` mode!"
                # In lockstep mode we sample MultiAgentBatches
                return decompress_sample_block(self.replay_buffers[_ALL_POLICIES].sample(num_items, **kwargs))
            elif policy_id is not None:
                return self._sample_single_policy(policy_id, num_items, **kwargs)
            else:
                return self._sample_all_policies(num_items, **kwargs)

    def _maybe_split_into_policy_batches(self, batch: SampleBatchType):
        """Returns a dict of policy IDs and batches, depending on our replay mode.

        This method helps with splitting up MultiAgentBatches only if the
        self.replay_mode requires it.
        """
        return batch.policy_batches

    @DeveloperAPI
    def _store(self):
        return self.replay_buffers

    def _sample_single_policy(self, policy_id, num_items, **kwargs):
        samples = []
        for turn in range(0, num_items):
            samples.append(decompress_sample_block(self.replay_buffers[policy_id].sample(1, **kwargs)))
        return concat_samples(samples)

    def _sample_all_policies(self, num_items, **kwargs):
        samples_all = {}
        for policy_id, replay_buffer in self.replay_buffers.items():
            samples_all[policy_id] = self._sample_single_policy(policy_id, num_items, **kwargs)
        return MultiAgentBatch(samples_all, sum(s.count for s in samples_all.values()))
