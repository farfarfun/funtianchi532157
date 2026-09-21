from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def test_steps_rejects_unknown_step() -> None:
    result = subprocess.run(
        ["bash", "funtianchi/process/steps.sh", "unknown"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "用法" in result.stderr
