# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise an installed AgentX package against a local streaming chat endpoint.

Run with the installed environment's Python, outside the source checkout if desired:
``python smoke_agentx_package.py --executable /venv/bin/agentx
--expected-version 1.0.7 --output-dir /tmp/agentx-smoke``.
No model, GPU, external server, Hugging Face download, or test extras are required.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any

import orjson


def prepare_inputs(root: Path) -> tuple[str, Path]:
    """Materialize a small local tokenizer and a four-turn Weka trajectory."""
    tokenizer_name = "agentx-smoke/local-wordlevel"
    snapshot = "0" * 40
    # AIPerf's offline loader resolves cached repo IDs rather than local paths.
    cache = root / "hf-cache" / "hub" / "models--agentx-smoke--local-wordlevel"
    tokenizer = cache / "snapshots" / snapshot
    tokenizer.mkdir(parents=True)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text(snapshot, encoding="utf-8")
    vocab = {"[UNK]": 0, **{f"word{i}": i + 1 for i in range(512)}}
    (tokenizer / "tokenizer.json").write_bytes(
        orjson.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
            }
        )
    )
    (tokenizer / "tokenizer_config.json").write_bytes(
        orjson.dumps(
            {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}
        )
    )
    traces = root / "traces"
    traces.mkdir()
    (traces / "smoke.json").write_bytes(
        orjson.dumps(
            {
                "id": "package-smoke",
                "models": ["mock-model"],
                "block_size": 16,
                "hash_id_scope": "local",
                "requests": [
                    {
                        "t": index * 0.1,
                        "type": "n",
                        "model": "mock-model",
                        "in": (index + 1) * 16 + 8,
                        "out": 2,
                        "hash_ids": list(range(1, index + 2)),
                        "input_types": ["text"],
                        "output_types": ["text"],
                        "stop": "end_turn",
                        "api_time": 0.05,
                        "think_time": 0.05 if index else 0.0,
                    }
                    for index in range(4)
                ],
            }
        )
    )
    return tokenizer_name, traces


class ChatEndpoint:
    """Serve bounded OpenAI-compatible SSE responses and retain received payloads."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.errors: list[str] = []

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Answer a single HTTP request and close the connection."""
        try:
            headers = (
                await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            ).decode()
            request_line, *header_lines = headers.split("\r\n")
            header_map = {
                key.lower(): value.strip()
                for line in header_lines
                if ":" in line
                for key, value in [line.split(":", 1)]
            }
            if request_line.split(" ")[1] == "/metrics":
                writer.write(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
                return
            if request_line.startswith("GET "):
                body = orjson.dumps({"object": "list", "data": [{"id": "mock-model"}]})
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
                return
            if request_line != "POST /v1/chat/completions HTTP/1.1":
                raise ValueError(f"Unexpected request: {request_line}")
            payload = orjson.loads(
                await asyncio.wait_for(
                    reader.readexactly(int(header_map["content-length"])), 10
                )
            )
            if not payload.get("stream") or not payload.get("messages"):
                raise ValueError(f"Expected streaming chat payload: {payload}")
            if payload.get("model") != "mock-model" or not payload.get("ignore_eos"):
                raise ValueError(
                    f"Missing model or AgentX ignore_eos setting: {payload}"
                )
            self.requests.append(payload)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n"
            )
            output_tokens = min(payload.get("max_tokens", 2), 2)
            for index in range(output_tokens):
                chunk = {
                    "id": f"chatcmpl-smoke-{len(self.requests)}",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "mock-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "word1 "},
                            "finish_reason": "stop"
                            if index == output_tokens - 1
                            else None,
                        }
                    ],
                }
                writer.write(b"data: " + orjson.dumps(chunk) + b"\n\n")
                await writer.drain()
                await asyncio.sleep(0.025)
            usage = {
                "id": f"chatcmpl-smoke-{len(self.requests)}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "mock-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 24,
                    "completion_tokens": output_tokens,
                    "total_tokens": 24 + output_tokens,
                },
            }
            writer.write(b"data: " + orjson.dumps(usage) + b"\n\ndata: [DONE]\n\n")
            await writer.drain()
        except Exception as exc:
            self.errors.append(str(exc))
        finally:
            writer.close()
            await writer.wait_closed()


async def run_command(command: list[str], root: Path, timeout: float) -> str:
    """Run the installed CLI in an isolated directory with network downloads disabled."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AIPERF_", "PYTHONPATH", "PYTHONHOME"))
    }
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(root / "hf-cache"),
        XDG_CACHE_HOME=str(root / "cache"),
        NO_PROXY="127.0.0.1,localhost",
        TOKENIZERS_PARALLELISM="false",
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=root,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = await process.communicate()
        await asyncio.to_thread((root / "timeout.log").write_bytes, output)
        raise RuntimeError(
            f"CLI timed out after {timeout}s; see {root / 'timeout.log'}"
        ) from None
    text = output.decode(errors="replace")
    await asyncio.to_thread((root / "cli.log").write_text, text, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(
            f"CLI exited {process.returncode}; see {root / 'cli.log'}:\n{text[-6000:]}"
        )
    return text


def check_artifacts(root: Path, endpoint: ChatEndpoint) -> int:
    """Verify that real requests completed and were exported under the AgentX scenario."""
    summary_paths = list((root / "artifacts").rglob("*profile_export_aiperf.json"))
    records_paths = list((root / "artifacts").rglob("*profile_export.jsonl"))
    assert len(summary_paths) == 1, f"Expected one summary, found {summary_paths}"
    assert len(records_paths) == 1, (
        f"Expected one records export, found {records_paths}"
    )
    summary = orjson.loads(summary_paths[0].read_bytes())
    records = [
        orjson.loads(line)
        for line in records_paths[0].read_bytes().splitlines()
        if line
    ]
    assert summary["input_config"]["scenario"] == "agentx", summary["input_config"]
    assert not summary.get("was_cancelled"), summary
    assert not summary.get("error_summary"), summary.get("error_summary")
    assert summary.get("error_request_count", {}).get("avg", 0) == 0, summary.get(
        "error_request_count"
    )
    profiling = [r for r in records if r["metadata"]["benchmark_phase"] == "profiling"]
    assert len(profiling) == 3, (
        f"Expected three profiled requests, got {len(profiling)}"
    )
    assert summary["request_count"]["avg"] == len(profiling), summary["request_count"]
    assert all(not record.get("error") for record in records), records
    assert all(
        record["metadata"]["source_trace_id"] == "package-smoke" for record in profiling
    ), profiling
    assert not endpoint.errors, endpoint.errors
    assert len(endpoint.requests) == len(records) == 4, (
        f"Expected four requests, got {len(endpoint.requests)} HTTP and {len(records)} records"
    )
    assert summary["output_token_count"]["sum"] > 0, summary.get("output_token_count")
    return len(profiling)


async def main() -> None:
    """Run and validate a three-second installed-package benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", default="agentx", help="Installed agentx command"
    )
    parser.add_argument(
        "--expected-version", help="Assert the installed release version"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New directory for logs and artifacts"
    )
    args = parser.parse_args()
    executable = shutil.which(args.executable)
    if executable is None:
        raise FileNotFoundError(f"Installed command not found: {args.executable}")
    executable = str(Path(executable).absolute())
    if args.output_dir:
        root = args.output_dir.resolve()
        await asyncio.to_thread(root.mkdir, parents=True)
    else:
        root = Path(
            await asyncio.to_thread(tempfile.mkdtemp, prefix="agentx-package-smoke-")
        )
    print(f"Smoke artifacts: {root}", flush=True)
    if args.expected_version:
        version = await run_command([executable, "--version"], root, 30)
        assert version.strip() == args.expected_version, version
    tokenizer, traces = await asyncio.to_thread(prepare_inputs, root)
    endpoint = ChatEndpoint()
    server = await asyncio.start_server(endpoint.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    command = [
        executable,
        "profile",
        "--unsafe-override",
        "--model",
        "mock-model",
        "--tokenizer",
        str(tokenizer),
        "--url",
        f"http://127.0.0.1:{port}",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--extra-inputs",
        "ignore_eos:true",
        "--custom-dataset-type",
        "weka_trace",
        "--input-file",
        str(traces),
        "--conversation-num",
        "1",
        "--no-fixed-schedule",
        "--concurrency",
        "1",
        "--benchmark-duration",
        "3",
        "--benchmark-grace-period",
        "5",
        "--warmup-requests-per-lane",
        "1",
        "--trajectory-start-min-ratio",
        "0",
        "--trajectory-start-max-ratio",
        "0",
        "--workers-max",
        "1",
        "--record-processors",
        "1",
        "--no-gpu-telemetry",
        "--ui",
        "simple",
        "--artifact-dir",
        str(root / "artifacts"),
    ]
    async with server:
        await run_command(command, root, 180)
    count = await asyncio.to_thread(check_artifacts, root, endpoint)
    print(
        f"PASS: installed agentx defaulted to scenario agentx; {count} profiling requests, "
        f"{len(endpoint.requests)} HTTP requests, zero errors. Artifacts: {root}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
