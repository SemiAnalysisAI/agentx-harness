# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.dataset.loader._weka_replay_timing import (
    compute_branch_start_ms,
    compute_delay_ms,
    think_ms_disagrees,
)


class TestComputeDelayMs:
    def test_normal_case_with_known_api_time(self):
        # prev request started at 1000ms, took 200ms (ends at 1200ms), next
        # request starts at 1500ms -> 300ms idle gap.
        assert compute_delay_ms(1000.0, 200.0, 1500.0) == pytest.approx(300.0)

    def test_none_api_time_degrades_to_start_to_start_gap(self):
        # v3 carries no durations: delay becomes the raw start-to-start gap.
        assert compute_delay_ms(1000.0, None, 1500.0) == pytest.approx(500.0)

    def test_zero_api_time_is_same_as_none(self):
        assert compute_delay_ms(1000.0, 0.0, 1500.0) == pytest.approx(500.0)

    def test_overlap_clamped_to_zero(self):
        # next request starts before the previous one's recorded end.
        assert compute_delay_ms(1000.0, 800.0, 1500.0) == 0.0

    def test_exact_boundary_is_zero(self):
        assert compute_delay_ms(1000.0, 500.0, 1500.0) == 0.0


class TestComputeBranchStartMs:
    def test_normal_case_children_start_after_parent_ends(self):
        # parent: starts 1000ms, takes 200ms -> ends 1200ms. Children start
        # at 1300/1400ms, both after the parent ends, so the branch starts
        # when the parent ends: 1200ms.
        result = compute_branch_start_ms(1000.0, 200.0, [1400.0, 1300.0])
        assert result == pytest.approx(1200.0)

    def test_none_api_time_degraded_case_uses_earliest_child(self):
        # api_time unknown (v3): parent_end degrades to the earliest child,
        # so the branch start is simply the earliest child timestamp.
        result = compute_branch_start_ms(1000.0, None, [1500.0, 1600.0])
        assert result == pytest.approx(1500.0)

    def test_overlap_child_starts_before_parent_response_finishes(self):
        # parent ends at 1200ms, but a child starts earlier, at 1100ms - the
        # real overlap must be reflected, not clamped to the parent's end.
        result = compute_branch_start_ms(1000.0, 200.0, [1100.0, 1400.0])
        assert result == pytest.approx(1100.0)

    def test_no_overlap_child_starts_well_after_parent_ends(self):
        result = compute_branch_start_ms(1000.0, 200.0, [5000.0])
        assert result == pytest.approx(1200.0)

    def test_empty_children_returns_none(self):
        assert compute_branch_start_ms(1000.0, 200.0, []) is None


class TestThinkMsDisagrees:
    def test_exact_match_agrees(self):
        assert think_ms_disagrees(1000.0, 1000.0) is False

    def test_small_absolute_difference_agrees(self):
        # 100ms off a 1000ms reference: well under both the 500ms floor and
        # the 10% relative band.
        assert think_ms_disagrees(1100.0, 1000.0) is False

    def test_large_absolute_difference_on_small_reference_disagrees(self):
        # 600ms off a tiny (10ms) reference: the 500ms absolute floor is what
        # catches this, not the relative band (10% of 10ms is 1ms).
        assert think_ms_disagrees(610.0, 10.0) is True

    def test_small_absolute_but_large_relative_difference_disagrees(self):
        # 20% off a large (100,000ms) reference: comfortably over the 500ms
        # floor, caught by the 10% relative band.
        assert think_ms_disagrees(120_000.0, 100_000.0) is True

    def test_just_under_threshold_agrees(self):
        # 10% of 100,000ms is 10,000ms; 9,999ms under agrees.
        assert think_ms_disagrees(109_999.0, 100_000.0) is False

    def test_just_over_threshold_disagrees(self):
        assert think_ms_disagrees(110_001.0, 100_000.0) is True

    def test_negative_direction_disagreement(self):
        assert think_ms_disagrees(100.0, 1000.0) is True
