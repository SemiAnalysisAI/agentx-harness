# SPDX-License-Identifier: Apache-2.0
"""A failed simulator must end the run without publishing phase success."""

import asyncio
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.common.bootstrap import _exit_if_service_failed
from aiperf.common.enums import CreditPhase, LifecycleState, ProfileCancelReason
from aiperf.common.environment import Environment
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    CommandSuccessResponse,
    ProfileCancelCommand,
)
from aiperf.common.mixins.command_handler_mixin import CommandHandlerMixin
from aiperf.common.models import (
    CreditPhaseStats,
    ErrorDetails,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker
from aiperf.simulation.protocol import SimulationError
from aiperf.timing.manager import TimingManager


def failure(*, abort_run: bool = True) -> BaseServiceErrorMessage:
    return BaseServiceErrorMessage(
        service_id="timing",
        error=ErrorDetails(type="SimulationError", message="invalid timing snapshot"),
        abort_run=abort_run,
    )


@pytest.mark.asyncio
async def test_driver_failure_collects_partial_records_and_exits_nonzero(
    system_controller: SystemController, benchmark_run: BenchmarkRun
) -> None:
    controller = system_controller
    controller.service_manager.service_id_map = {
        "records": SimpleNamespace(service_type=ServiceType.RECORDS_MANAGER),
    }
    partial = ProcessRecordsResult(
        results=ProfileResults(
            records=[],
            completed=0,
            start_ns=1,
            end_ns=2,
            error_request_count=1,
            was_cancelled=True,
        )
    )
    records = RecordsManager.__new__(RecordsManager)
    records.warning = Mock()
    records.debug = Mock()
    records._records_tracker = RecordsTracker()
    records._records_tracker.update_phase_info(
        CreditPhaseStats(phase=CreditPhase.PROFILING, phase_index=0)
    )
    records._process_results_lock = asyncio.Lock()
    records._processed_results = {}
    records._process_results_impl = AsyncMock(return_value=partial)
    cancel_received, release_records, error_published = (
        asyncio.Event() for _ in range(3)
    )
    deliveries = []
    commands = []

    async def deliver(command: ProfileCancelCommand) -> None:
        # Each bus message has its own callback task, as in ZMQSubClient.
        await controller._record_abort_on_profile_cancel(command)
        cancel_received.set()
        await release_records.wait()
        result = await records._on_profile_cancel_command(command)
        await CommandHandlerMixin._process_command_response_message(
            controller,
            CommandSuccessResponse.from_command_message(command, "records", result),
        )

    async def broadcast(command: ProfileCancelCommand) -> None:
        commands.append(command)
        deliveries.append(asyncio.create_task(deliver(command)))

    controller.publish = AsyncMock(side_effect=broadcast)
    manager = TimingManager(run=benchmark_run, service_id="timing")
    manager._phase_orchestrator = SimpleNamespace(
        start=AsyncMock(), state=LifecycleState.STOPPED
    )
    manager._simulator = SimpleNamespace(
        drive=AsyncMock(side_effect=SimulationError("invalid timing snapshot"))
    )
    manager._simulation_publisher = Mock(flush=AsyncMock())
    reports = []

    def schedule(coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        reports.append(task)
        return task

    async def report(message: BaseServiceErrorMessage) -> None:
        error_published.set()
        await controller._process_service_error_message(message)

    manager.execute_async = schedule
    manager.publish = AsyncMock(side_effect=report)
    phase_task = asyncio.create_task(manager._run_profiling())
    phase_task.add_done_callback(manager._on_phase_orchestrator_done)
    with pytest.raises(SimulationError, match="invalid timing snapshot"):
        await phase_task
    await asyncio.wait_for(error_published.wait(), 1)
    await asyncio.wait_for(cancel_received.wait(), 1)
    assert commands[0].reason == ProfileCancelReason.SERVICE_ERROR
    assert commands[0].reason_detail == manager.publish.call_args.args[0].error.message
    controller.stop.assert_not_awaited()
    release_records.set()
    await asyncio.wait_for(asyncio.gather(*reports, *deliveries), 1)

    manager._simulation_publisher.flush.assert_not_awaited()
    assert records._records_tracker.was_phase_cancelled(CreditPhase.PROFILING)
    records._process_results_impl.assert_awaited_once_with(CreditPhase.PROFILING, True)
    assert controller._profile_results is partial
    assert controller._profile_results.results.error_request_count == 1
    controller.stop.assert_awaited_once()
    assert len(controller._exit_errors) == 1
    assert controller._exit_errors[0].error_details.type == "SimulationError"
    with pytest.raises(SystemExit) as exit_info:
        _exit_if_service_failed(controller)
    assert exit_info.value.code == 1


@pytest.mark.asyncio
async def test_fatal_error_stops_after_records_response_timeout(
    system_controller: SystemController, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = system_controller
    controller.service_manager.service_id_map = {
        "records": SimpleNamespace(service_type=ServiceType.RECORDS_MANAGER),
    }
    controller.publish = AsyncMock()  # No command response will arrive.
    monkeypatch.setattr(Environment.SERVICE, "PROFILE_CANCEL_TIMEOUT", 0.01)
    await asyncio.wait_for(controller._process_service_error_message(failure()), 1)
    controller.stop.assert_awaited_once()
    assert controller._profile_results is None
    assert not controller._multi_response_futures
    assert controller._exit_errors[0].error_details.message == "invalid timing snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_cancel_first", [False, True])
async def test_fatal_error_during_existing_cancel_does_not_restart_shutdown(
    system_controller: SystemController, user_cancel_first: bool
) -> None:
    controller = system_controller
    controller.service_manager.service_id_map = {
        "records": SimpleNamespace(service_type=ServiceType.RECORDS_MANAGER),
    }
    command_sent = asyncio.Event()
    commands = []

    async def broadcast(command: ProfileCancelCommand) -> None:
        commands.append(command)
        command_sent.set()

    controller.publish = AsyncMock(side_effect=broadcast)
    first = asyncio.create_task(
        controller._cancel_profiling()
        if user_cancel_first
        else controller._process_service_error_message(failure())
    )
    await command_sent.wait()
    await controller._process_service_error_message(failure())
    assert len(commands) == 1
    await CommandHandlerMixin._process_command_response_message(
        controller, CommandSuccessResponse.from_command_message(commands[0], "records")
    )
    await asyncio.wait_for(first, 1)
    controller.stop.assert_awaited_once()
    assert controller._exit_errors


@pytest.mark.asyncio
async def test_default_service_error_retains_existing_completion_behavior(
    system_controller: SystemController,
) -> None:
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._cancel_profiling = AsyncMock()
    message = BaseServiceErrorMessage(
        service_id="ordinary-service", error=ErrorDetails(message="ordinary failure")
    )
    assert not message.abort_run
    await controller._process_service_error_message(message)
    controller._check_and_trigger_shutdown.assert_awaited_once()
    controller._cancel_profiling.assert_not_awaited()


@pytest.mark.parametrize("simulator", [False, True])
@pytest.mark.parametrize("from_details", [False, True])
def test_only_simulator_phase_failures_request_run_abort(
    benchmark_run: BenchmarkRun, simulator: bool, from_details: bool
) -> None:
    manager = TimingManager(run=benchmark_run, service_id="timing")
    manager._simulator = Mock() if simulator else None
    manager._publish_service_error_safely = Mock()
    if from_details:
        manager._publish_phase_failure_from_details(
            ErrorDetails(message="phase failed")
        )
    else:
        manager._publish_phase_failure(RuntimeError("phase failed"))
    message = manager._publish_service_error_safely.call_args.args[0]
    assert message.abort_run is simulator
