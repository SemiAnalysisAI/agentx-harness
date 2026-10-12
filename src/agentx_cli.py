# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX command-line branding and defaults over the upstream AIPerf CLI."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from cyclopts import App

from aiperf.cli import app as aiperf_app


def _version() -> str:
    """Read the AgentX distribution version independently of upstream AIPerf."""
    try:
        return version("agentx-harness")
    except PackageNotFoundError:
        return "unknown"


app = App(
    name="agentx",
    version=_version,
    help=(
        "AgentX Harness - agentic inference benchmarking, powered by AIPerf.\n\n"
        "The profile command defaults to --scenario agentx when using CLI flags. "
        "An explicit --scenario takes precedence. With --config/-f, the YAML "
        "configuration supplies the scenario."
    ),
)
app.update(aiperf_app)
# Copied root commands are bound to AIPerf; bind these to the branded app.
app.help_flags = ("--help", "-h")
app.version_flags = ("--version",)
del app["--install-completion"]
app.register_install_completion_command()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the upstream commands with AgentX defaults for flag-based profiles."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "profile":
        options = tokens[1 : tokens.index("--") if "--" in tokens else len(tokens)]
        has_scenario_or_config = any(
            token in {"--scenario", "--config", "-f"}
            or token.startswith(("--scenario=", "--config=", "-f"))
            for token in options
        )
        if not has_scenario_or_config:
            tokens[1:1] = ["--scenario", "agentx"]
    app(tokens)


if __name__ == "__main__":
    main()
