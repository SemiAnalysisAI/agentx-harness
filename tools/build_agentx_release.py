# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the AgentX distribution without changing upstream's source metadata.

Run with ``uv run --no-project --with tomlkit --with packaging
tools/build_agentx_release.py --tag agentx-v1.0.7``.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import tomlkit
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def release_version(tag: str) -> str:
    """Return the canonical PyPI version from an AgentX release tag."""
    if not tag.startswith("agentx-v"):
        raise ValueError("Release tags must start with agentx-v")
    value = tag.removeprefix("agentx-v")
    version = Version(value)
    if str(version) != value or version.local is not None or version.epoch:
        raise ValueError(f"Use a canonical public version in the release tag: {tag}")
    return value


def prepare_source(source: Path, destination: Path, version: str) -> None:
    """Copy tracked files and apply distribution-only metadata in isolation."""
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=source)
        .decode()
        .split("\0")
    )
    for filename in filter(None, tracked):
        original = source / filename
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target, follow_symlinks=False)

    path = destination / "pyproject.toml"
    metadata = tomlkit.parse(path.read_text())
    project = metadata["project"]
    if project["name"] != "aiperf":
        raise ValueError("Expected upstream aiperf source metadata")
    project["name"] = "agentx-harness"
    project["version"] = version
    project["description"] = (
        "SemiAnalysis AgentX inference benchmark harness, built on AIPerf"
    )
    project["scripts"]["agentx"] = "agentx_cli:main"
    project["urls"] = {
        "Homepage": "https://github.com/SemiAnalysisAI/agentx-harness",
        "Source": "https://github.com/SemiAnalysisAI/agentx-harness",
        "Issues": "https://github.com/SemiAnalysisAI/agentx-harness/issues",
    }
    # The public AgentX corpora need datasets' heterogeneous JSON support.
    for index, dependency in enumerate(project["dependencies"]):
        requirement = Requirement(dependency)
        if requirement.name == "datasets":
            requirement.specifier &= SpecifierSet(">=4.7.0")
            project["dependencies"][index] = str(requirement)
    for dependencies in project["optional-dependencies"].values():
        for index, dependency in enumerate(dependencies):
            if dependency.startswith("aiperf["):
                dependencies[index] = dependency.replace(
                    "aiperf[", "agentx-harness[", 1
                )
    metadata["tool"]["hatch"]["build"] = {
        "artifacts": ["src/aiperf/_build_info.py"],
        "targets": {
            "wheel": {
                "packages": ["src/aiperf"],
                "force-include": {"src/agentx_cli.py": "agentx_cli.py"},
            }
        },
    }
    path.write_text(tomlkit.dumps(metadata))

    init_path = destination / "src/aiperf/__init__.py"
    init_text = init_path.read_text()
    if init_text.count('version("aiperf")') != 1:
        raise ValueError("Upstream version lookup changed; review the release adapter")
    init_path.write_text(
        init_text.replace('version("aiperf")', 'version("agentx-harness")')
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    (destination / "src/aiperf/_build_info.py").write_text(f'COMMIT_SHA = "{commit}"\n')


def main() -> None:
    """Build a wheel and sdist using the requested GitHub release version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    version = release_version(args.tag)
    source = Path(__file__).resolve().parents[1]
    output = args.out_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="agentx-release-") as temporary:
        destination = Path(temporary)
        prepare_source(source, destination, version)
        subprocess.run(
            ["uv", "build", "--out-dir", str(output), str(destination)], check=True
        )


if __name__ == "__main__":
    main()
