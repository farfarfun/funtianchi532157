import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def test_steps_rejects_unknown_step() -> None:
    result = subprocess.run(
        ["bash", "funtianchi532157/process/steps.sh", "unknown"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "用法" in result.stderr


def test_step31_resolves_paths_from_script_location(tmp_path: Path) -> None:
    """数据处理步骤不应依赖调用者的当前目录。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_file = tmp_path / "calls"
    python_stub = bin_dir / "python"
    python_stub.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CALLS_FILE"\n',
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    env = os.environ | {
        "CALLS_FILE": str(calls_file),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }

    subprocess.run(
        ["bash", str(ROOT / "funtianchi532157/process/steps.sh"), "step31"],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    calls = calls_file.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert all(
        str(ROOT / "funtianchi532157/process/process_data.py") in call for call in calls
    )
    assert "alpaca-cot-en-refine.yaml" in calls[0]
    assert "alpaca-cot-zh-refine.yaml" in calls[1]


def test_prepare_data_resolves_paths_from_script_location(tmp_path: Path) -> None:
    """数据下载脚本应将数据保存到脚本所在项目目录。"""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    script = project_dir / "prepare_data_and_models.sh"
    shutil.copy2(ROOT / "funtianchi532157/prepare_data_and_models.sh", script)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_file = tmp_path / "calls"
    for command in ("funget", "tar"):
        stub = bin_dir / command
        stub.write_text(
            '#!/bin/sh\nprintf \'%s\\n\' "$PWD" >> "$CALLS_FILE"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
    env = os.environ | {
        "CALLS_FILE": str(calls_file),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }

    subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, check=True)

    calls = calls_file.read_text(encoding="utf-8").splitlines()
    assert calls
    assert all(call.startswith(str(project_dir / "data")) for call in calls)


def test_training_command_uses_expected_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """训练步骤应调用仓库中实际存在的训练脚本。"""
    from funtianchi532157 import train

    calls: list[list[str]] = []

    def record(command: list[str], check: bool) -> None:
        assert check is True
        calls.append(command)

    monkeypatch.setattr(train.subprocess, "run", record)
    train.step41()

    assert calls == [
        [
            "bash",
            str(ROOT / "funtianchi532157/lm-training/train_scripts/1b-T4/train.sh"),
            str(ROOT / "funtianchi532157/data/models/falcon-rw-1b"),
            str(ROOT / "funtianchi532157/outputs/refined_data/en_refine.jsonl"),
            str(ROOT / "funtianchi532157/outputs/finetuned_model"),
        ]
    ]


def test_training_failure_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    """训练子进程失败时不应吞掉异常。"""
    from funtianchi532157 import train

    def fail(command: list[str], check: bool) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(train.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        train.step41()


def test_step31_uses_language_specific_refine_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中英文数据清洗必须分别使用各自的 refine 配置，不能共用英文配置。"""
    from funtianchi532157 import train

    calls: list[list[str]] = []

    def record(command: list[str], check: bool) -> None:
        assert check is True
        calls.append(command)

    monkeypatch.setattr(train.subprocess, "run", record)
    train.step31()

    assert len(calls) == 2
    assert "alpaca-cot-en-refine.yaml" in " ".join(calls[0])
    assert "raw_data_en.jsonl" in " ".join(calls[0])
    assert "alpaca-cot-zh-refine.yaml" in " ".join(calls[1])
    assert "raw_data_zh.jsonl" in " ".join(calls[1])


def test_evaluation_command_uses_project_relative_challenge_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """评估步骤的挑战数据路径必须基于仓库自身位置，不能硬编码成固定绝对路径。"""
    from funtianchi532157 import train

    calls: list[list[str]] = []

    def record(command: list[str], check: bool, cwd: str) -> None:
        assert check is True
        assert cwd == str(ROOT / "funtianchi532157")
        calls.append(command)

    monkeypatch.setattr(train.subprocess, "run", record)
    train.step51()

    assert calls == [
        [
            "bash",
            str(
                ROOT
                / "funtianchi532157/lm-evaluation-harness/examples/challenge-1B-stage1.sh"
            ),
            "dev",
            str(ROOT / "funtianchi532157/outputs/finetuned_model"),
            str(ROOT / "funtianchi532157/data/challenge-data"),
            str(ROOT / "funtianchi532157/outputs/eval_dev_results"),
        ]
    ]


def test_evaluation_failure_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    """评估子进程失败时不应吞掉异常。"""
    from funtianchi532157 import train

    def fail(command: list[str], check: bool, cwd: str) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(train.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        train.step51()


def test_process_data_runs_default_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认数据处理执行器应被创建并运行。"""
    from funtianchi532157.process import process_data

    ran: list[bool] = []

    class FakeExecutor:
        def __init__(self, config: SimpleNamespace) -> None:
            assert config.executor_type == "default"

        def run(self) -> None:
            ran.append(True)

    monkeypatch.setattr(
        process_data, "init_configs", lambda: SimpleNamespace(executor_type="default")
    )
    monkeypatch.setattr(process_data, "DefaultExecutor", FakeExecutor)

    process_data.main()

    assert ran == [True]


def test_process_data_rejects_unknown_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未知执行器类型应提供明确错误。"""
    from funtianchi532157.process import process_data

    monkeypatch.setattr(
        process_data, "init_configs", lambda: SimpleNamespace(executor_type="unknown")
    )

    with pytest.raises(ValueError, match="unknown"):
        process_data.main()
