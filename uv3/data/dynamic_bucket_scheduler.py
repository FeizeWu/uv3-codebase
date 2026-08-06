"""Pure state machine for supply-driven distributed text-bucket scheduling."""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


def smooth_weighted_schedule(
    buckets: Sequence[int], weights: Sequence[int],
) -> tuple[int, ...]:
    """Build a deterministic low-discrepancy schedule, allowing zero weights."""
    buckets = tuple(int(bucket) for bucket in buckets)
    weights = tuple(int(weight) for weight in weights)
    if (
        not buckets
        or len(buckets) != len(weights)
        or any(weight < 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError(f"invalid smooth schedule: buckets={buckets}, weights={weights}")
    total = sum(weights)
    scores = [0] * len(buckets)
    schedule = []
    for _ in range(total):
        for index, weight in enumerate(weights):
            scores[index] += weight
        chosen = max(range(len(buckets)), key=lambda index: scores[index])
        scores[chosen] -= total
        schedule.append(buckets[chosen])
    return tuple(schedule)


def ready_mask_from_counts(
    eligible_counts_by_resolution: Sequence[Sequence[int]],
    batch_size: int,
) -> int:
    """Return the monotone target-ready bitmask for rank-local queue counts.

    Every row contains native bucket counts for one resolution. Target ``i``
    accepts the cumulative native counts ``0..i``.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = [tuple(int(value) for value in row) for row in eligible_counts_by_resolution]
    if not rows:
        raise ValueError("at least one resolution is required")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("all resolution count rows must have equal non-zero width")
    mask = 0
    for target_index in range(width):
        if any(sum(row[: target_index + 1]) >= batch_size for row in rows):
            mask |= 1 << target_index
    # Readiness must be a suffix: eligibility only grows with target length.
    seen_ready = False
    for index in range(width):
        ready = bool(mask & (1 << index))
        if seen_ready and not ready:
            raise AssertionError(f"non-monotone ready mask: {mask:b}")
        seen_ready |= ready
    return mask


def select_shortest_common_target(
    buckets: Sequence[int], desired_target: int, common_ready_mask: int,
) -> int:
    """Choose the shortest globally ready target that is not below desired."""
    buckets = tuple(int(bucket) for bucket in buckets)
    try:
        desired_index = buckets.index(int(desired_target))
    except ValueError as error:
        raise ValueError(f"desired target {desired_target} is not in {buckets}") from error
    for index in range(desired_index, len(buckets)):
        if common_ready_mask & (1 << index):
            return buckets[index]
    raise RuntimeError(
        f"no common ready target >= {desired_target}; mask={common_ready_mask:#x}"
    )


def safe_integer_weights(
    worst_rank_native_counts: Sequence[int],
    *,
    safety_margin: float,
    total_weight: int = 100,
) -> tuple[int, ...]:
    """Convert worst-rank supply to integer demand without crossing safe CDF.

    ``worst_rank_native_counts`` are the native counts of the rank/window that
    defines each prefix minimum. They need not come from one physical rank;
    callers may all-reduce prefix counts with MIN and pass their differences.
    """
    counts = tuple(int(value) for value in worst_rank_native_counts)
    total_weight = int(total_weight)
    if not counts or any(value < 0 for value in counts) or sum(counts) <= 0:
        raise ValueError(f"invalid native counts: {counts}")
    if total_weight < 1:
        raise ValueError("total_weight must be positive")
    if not 0.0 <= float(safety_margin) < 1.0:
        raise ValueError("safety_margin must be in [0,1)")
    samples = sum(counts)
    prefix_weights = []
    cumulative = 0
    previous = 0
    for count in counts[:-1]:
        cumulative += count
        safe_fraction = max(0.0, cumulative / samples - float(safety_margin))
        demand = math.floor(total_weight * safe_fraction + 1e-12)
        demand = min(total_weight, max(previous, demand))
        prefix_weights.append(demand)
        previous = demand
    boundaries = (0, *prefix_weights, total_weight)
    weights = tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(len(counts))
    )
    if sum(weights) != total_weight or any(weight < 0 for weight in weights):
        raise AssertionError(f"invalid derived weights: {weights}")
    return weights


@dataclass(frozen=True)
class SchedulerSelection:
    desired_target: int
    selected_target: int
    fallback: bool
    fallback_to_max: bool
    extra_text_tokens_per_sample: int


@dataclass(frozen=True)
class BucketControlResult:
    common_ready_mask: int
    all_have_completed_window: bool
    any_lookahead_exhausted: bool
    wait_ms: float


class LocalBucketController:
    """No-collective controller used by unit tests and one-rank smoke runs."""

    rank = 0
    world_size = 1

    def synchronize(
        self,
        local_ready_mask: int,
        *,
        has_completed_window: bool,
        lookahead_exhausted: bool,
    ) -> BucketControlResult:
        return BucketControlResult(
            common_ready_mask=int(local_ready_mask),
            all_have_completed_window=bool(has_completed_window),
            any_lookahead_exhausted=bool(lookahead_exhausted),
            wait_ms=0.0,
        )

    def update_weights(
        self,
        local_native_counts: Sequence[int],
        *,
        old_weights: Sequence[int],
        schedule_version: int,
        safety_margin: float,
    ) -> tuple[tuple[int, ...], dict, float]:
        counts = tuple(int(value) for value in local_native_counts)
        weights = safe_integer_weights(counts, safety_margin=safety_margin)
        samples = sum(counts)
        cumulative = 0
        cdf = []
        for count in counts[:-1]:
            cumulative += count
            cdf.append(cumulative / samples)
        event = {
            "event": "bucket_schedule_update",
            "schedule_version": int(schedule_version) + 1,
            "samples_per_rank_min": samples,
            "samples_per_rank_max": samples,
            "worst_rank_supply_cdf": cdf,
            "old_weights": [int(value) for value in old_weights],
            "new_weights": list(weights),
            "safety_margin": float(safety_margin),
        }
        return weights, event, 0.0


class DistributedBucketController:
    """Tiny fixed-order Gloo collectives for globally consistent text targets."""

    def __init__(self, process_group, bucket_count: int) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("distributed bucket control requires initialized torch.distributed")
        self.process_group = process_group
        self.bucket_count = int(bucket_count)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def synchronize(
        self,
        local_ready_mask: int,
        *,
        has_completed_window: bool,
        lookahead_exhausted: bool,
    ) -> BucketControlResult:
        import time
        import torch.distributed as dist

        has_window_bit = 1 << self.bucket_count
        no_exhaustion_bit = 1 << (self.bucket_count + 1)
        word = int(local_ready_mask)
        if has_completed_window:
            word |= has_window_bit
        if not lookahead_exhausted:
            word |= no_exhaustion_bit
        value = torch.tensor([word], dtype=torch.int64, device="cpu")
        started = time.perf_counter()
        dist.all_reduce(value, op=dist.ReduceOp.BAND, group=self.process_group)
        wait_ms = (time.perf_counter() - started) * 1000.0
        common = int(value.item())
        ready_bits = (1 << self.bucket_count) - 1
        return BucketControlResult(
            common_ready_mask=common & ready_bits,
            all_have_completed_window=bool(common & has_window_bit),
            any_lookahead_exhausted=not bool(common & no_exhaustion_bit),
            wait_ms=wait_ms,
        )

    def update_weights(
        self,
        local_native_counts: Sequence[int],
        *,
        old_weights: Sequence[int],
        schedule_version: int,
        safety_margin: float,
    ) -> tuple[tuple[int, ...], dict, float]:
        import time
        import torch.distributed as dist

        counts = tuple(int(value) for value in local_native_counts)
        samples = sum(counts)
        prefix = []
        cumulative = 0
        for count in counts[:-1]:
            cumulative += count
            prefix.append(cumulative)
        prefix_tensor = torch.tensor(prefix, dtype=torch.int64, device="cpu")
        sample_min = torch.tensor([samples], dtype=torch.int64, device="cpu")
        sample_max = sample_min.clone()
        started = time.perf_counter()
        dist.all_reduce(prefix_tensor, op=dist.ReduceOp.MIN, group=self.process_group)
        dist.all_reduce(sample_min, op=dist.ReduceOp.MIN, group=self.process_group)
        dist.all_reduce(sample_max, op=dist.ReduceOp.MAX, group=self.process_group)
        minimum = int(sample_min.item())
        maximum = int(sample_max.item())
        if minimum != maximum or minimum != samples:
            raise RuntimeError(
                "long-term bucket windows must contain the same exact sample count "
                f"on every rank; local={samples} min={minimum} max={maximum}"
            )
        update = torch.zeros(len(counts) + 1, dtype=torch.int64, device="cpu")
        worst_prefixes = tuple(int(value) for value in prefix_tensor.tolist())
        if self.rank == 0:
            synthetic_counts = prefix_minima_to_native_counts(worst_prefixes, minimum)
            weights = safe_integer_weights(
                synthetic_counts,
                safety_margin=safety_margin,
                total_weight=100,
            )
            update[0] = int(schedule_version) + 1
            update[1:] = torch.tensor(weights, dtype=torch.int64)
        dist.broadcast(update, src=0, group=self.process_group)
        wait_ms = (time.perf_counter() - started) * 1000.0
        received_version = int(update[0].item())
        if received_version != int(schedule_version) + 1:
            raise RuntimeError(
                f"invalid broadcast schedule version {received_version}; "
                f"expected {int(schedule_version) + 1}"
            )
        weights = tuple(int(value) for value in update[1:].tolist())
        event = {
            "event": "bucket_schedule_update",
            "schedule_version": received_version,
            "samples_per_rank_min": minimum,
            "samples_per_rank_max": maximum,
            "worst_rank_supply_cdf": [value / minimum for value in worst_prefixes],
            "old_weights": [int(value) for value in old_weights],
            "new_weights": list(weights),
            "safety_margin": float(safety_margin),
        }
        return weights, event, wait_ms


class DynamicJointBucketScheduler:
    """Checkpointable pure scheduler; all distributed I/O lives in a controller."""

    STATE_VERSION = 1

    def __init__(
        self,
        buckets: Sequence[int],
        initial_weights: Sequence[int],
        *,
        batch_size: int,
        resolution_count: int,
        long_term_window_per_rank: int = 50_000,
        safety_margin: float = 0.02,
    ) -> None:
        self.buckets = tuple(int(bucket) for bucket in buckets)
        self.initial_weights = tuple(int(weight) for weight in initial_weights)
        self.batch_size = int(batch_size)
        self.resolution_count = int(resolution_count)
        self.long_term_window_per_rank = int(long_term_window_per_rank)
        self.safety_margin = float(safety_margin)
        if not self.buckets or tuple(sorted(self.buckets)) != self.buckets:
            raise ValueError(f"buckets must be non-empty and sorted: {self.buckets}")
        if len(set(self.buckets)) != len(self.buckets):
            raise ValueError(f"buckets must be unique: {self.buckets}")
        if self.batch_size < 1 or self.resolution_count < 1:
            raise ValueError("batch_size and resolution_count must be positive")
        if self.long_term_window_per_rank < 1:
            raise ValueError("long_term_window_per_rank must be positive")
        if not 0.0 <= self.safety_margin < 1.0:
            raise ValueError("safety_margin must be in [0,1)")
        self.base_weights = self.initial_weights
        self.schedule = smooth_weighted_schedule(self.buckets, self.base_weights)
        self.schedule_cursor = 0
        self.schedule_version = 0
        self._current_native_counts = [0] * len(self.buckets)
        self._current_window_samples = 0
        self._completed_windows: deque[tuple[int, ...]] = deque()
        self.desired_target_counts = Counter({bucket: 0 for bucket in self.buckets})
        self.selected_target_counts = Counter({bucket: 0 for bucket in self.buckets})
        self.fallback_counts = Counter()
        self.fallback_steps = 0
        self.fallback_to_max_steps = 0
        self.fallback_extra_text_tokens_sum = 0

    @property
    def minimum_ready_buffer(self) -> int:
        return self.resolution_count * (self.batch_size - 1) + 1

    @property
    def desired_target(self) -> int:
        return self.schedule[self.schedule_cursor % len(self.schedule)]

    @property
    def at_schedule_boundary(self) -> bool:
        return self.schedule_cursor % len(self.schedule) == 0

    @property
    def has_completed_window(self) -> bool:
        return bool(self._completed_windows)

    def observe_native_target(self, native_target: int) -> None:
        try:
            index = self.buckets.index(int(native_target))
        except ValueError as error:
            raise ValueError(f"native target {native_target} is not in {self.buckets}") from error
        self._current_native_counts[index] += 1
        self._current_window_samples += 1
        if self._current_window_samples == self.long_term_window_per_rank:
            self._completed_windows.append(tuple(self._current_native_counts))
            self._current_native_counts = [0] * len(self.buckets)
            self._current_window_samples = 0
        elif self._current_window_samples > self.long_term_window_per_rank:
            raise AssertionError("native-count window overflow")

    def peek_completed_window(self) -> tuple[int, ...]:
        if not self._completed_windows:
            raise RuntimeError("no completed native-count window")
        return self._completed_windows[0]

    def pop_completed_window(self) -> tuple[int, ...]:
        if not self._completed_windows:
            raise RuntimeError("no completed native-count window")
        return self._completed_windows.popleft()

    def ready_mask(self, native_counts_by_resolution: Sequence[Sequence[int]]) -> int:
        return ready_mask_from_counts(native_counts_by_resolution, self.batch_size)

    def select(self, common_ready_mask: int) -> SchedulerSelection:
        desired = self.desired_target
        selected = select_shortest_common_target(
            self.buckets, desired, int(common_ready_mask),
        )
        fallback = selected != desired
        return SchedulerSelection(
            desired_target=desired,
            selected_target=selected,
            fallback=fallback,
            fallback_to_max=fallback and selected == self.buckets[-1],
            extra_text_tokens_per_sample=max(0, selected - desired),
        )

    def record_selection(self, selection: SchedulerSelection) -> None:
        if selection.desired_target != self.desired_target:
            raise RuntimeError(
                f"selection desired={selection.desired_target} does not match "
                f"schedule desired={self.desired_target}"
            )
        self.desired_target_counts[selection.desired_target] += 1
        self.selected_target_counts[selection.selected_target] += 1
        if selection.fallback:
            self.fallback_steps += 1
            self.fallback_counts[
                (selection.desired_target, selection.selected_target)
            ] += 1
        if selection.fallback_to_max:
            self.fallback_to_max_steps += 1
        self.fallback_extra_text_tokens_sum += (
            selection.extra_text_tokens_per_sample * self.batch_size
        )
        self.schedule_cursor += 1

    def install_weights(self, weights: Sequence[int], schedule_version: int) -> None:
        weights = tuple(int(weight) for weight in weights)
        expected_version = self.schedule_version + 1
        if int(schedule_version) != expected_version:
            raise RuntimeError(
                f"schedule version must advance {self.schedule_version}->{expected_version}, "
                f"got {schedule_version}"
            )
        if not self.at_schedule_boundary:
            raise RuntimeError("new weights may only be installed at a schedule boundary")
        self.base_weights = weights
        self.schedule = smooth_weighted_schedule(self.buckets, weights)
        self.schedule_cursor = 0
        self.schedule_version = int(schedule_version)

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "buckets": self.buckets,
            "initial_weights": self.initial_weights,
            "base_weights": self.base_weights,
            "schedule": self.schedule,
            "schedule_cursor": self.schedule_cursor,
            "schedule_version": self.schedule_version,
            "batch_size": self.batch_size,
            "resolution_count": self.resolution_count,
            "long_term_window_per_rank": self.long_term_window_per_rank,
            "safety_margin": self.safety_margin,
            "current_native_counts": tuple(self._current_native_counts),
            "current_window_samples": self._current_window_samples,
            "completed_windows": tuple(self._completed_windows),
            "desired_target_counts": dict(self.desired_target_counts),
            "selected_target_counts": dict(self.selected_target_counts),
            "fallback_counts": dict(self.fallback_counts),
            "fallback_steps": self.fallback_steps,
            "fallback_to_max_steps": self.fallback_to_max_steps,
            "fallback_extra_text_tokens_sum": self.fallback_extra_text_tokens_sum,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state.get("version", 0)) != self.STATE_VERSION:
            raise ValueError(f"unsupported scheduler state version: {state.get('version')}")
        expected = {
            "buckets": self.buckets,
            "initial_weights": self.initial_weights,
            "batch_size": self.batch_size,
            "resolution_count": self.resolution_count,
            "long_term_window_per_rank": self.long_term_window_per_rank,
            "safety_margin": self.safety_margin,
        }
        for key, value in expected.items():
            saved = state.get(key)
            if key == "safety_margin":
                matches = float(saved) == float(value)
            elif isinstance(value, tuple):
                matches = tuple(saved or ()) == value
            else:
                matches = saved == value
            if not matches:
                raise ValueError(
                    f"scheduler resume mismatch for {key}: checkpoint={saved!r} "
                    f"current={value!r}"
                )
        self.base_weights = tuple(int(x) for x in state["base_weights"])
        self.schedule = tuple(int(x) for x in state["schedule"])
        if self.schedule != smooth_weighted_schedule(self.buckets, self.base_weights):
            raise ValueError("checkpoint schedule does not match its base weights")
        self.schedule_cursor = int(state["schedule_cursor"])
        self.schedule_version = int(state["schedule_version"])
        self._current_native_counts = [int(x) for x in state["current_native_counts"]]
        self._current_window_samples = int(state["current_window_samples"])
        if sum(self._current_native_counts) != self._current_window_samples:
            raise ValueError("current native-count window is internally inconsistent")
        self._completed_windows = deque(
            tuple(int(x) for x in counts) for counts in state["completed_windows"]
        )
        if any(sum(counts) != self.long_term_window_per_rank for counts in self._completed_windows):
            raise ValueError("completed native-count window has the wrong sample count")
        self.desired_target_counts = Counter({
            int(key): int(value)
            for key, value in state.get("desired_target_counts", {}).items()
        })
        self.selected_target_counts = Counter({
            int(key): int(value)
            for key, value in state.get("selected_target_counts", {}).items()
        })
        self.fallback_counts = Counter({
            tuple(int(x) for x in key): int(value)
            for key, value in state.get("fallback_counts", {}).items()
        })
        self.fallback_steps = int(state.get("fallback_steps", 0))
        self.fallback_to_max_steps = int(state.get("fallback_to_max_steps", 0))
        self.fallback_extra_text_tokens_sum = int(
            state.get("fallback_extra_text_tokens_sum", 0)
        )


def prefix_minima_to_native_counts(prefix_minima: Iterable[int], total: int) -> tuple[int, ...]:
    """Convert all-reduced prefix minima back to a synthetic native histogram."""
    prefixes = tuple(int(value) for value in prefix_minima)
    boundaries = (0, *prefixes, int(total))
    counts = tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    )
    if any(value < 0 for value in counts):
        raise ValueError(f"prefix minima are not monotone: {prefixes}")
    return counts
