"""Launcher tests.

Two tiers:
  - Always-on unit tests for the registry and the Docker ``docker run`` argv
    construction (pure, no daemon).
  - A real subprocess↔docker parity test that builds a slim image and runs the
    same probe both ways — skipped when Docker or the network is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mlprobe import (
    DockerLauncher,
    InProcessLauncher,
    SubprocessLauncher,
    get_launcher,
)
from mlprobe.launchers import ProbeLauncher
from mlprobe.probe import ProbeInput

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---- registry -------------------------------------------------------------

def test_get_launcher_by_name():
    assert isinstance(get_launcher("subprocess"), SubprocessLauncher)
    assert isinstance(get_launcher("in_process"), InProcessLauncher)


def test_get_launcher_passthrough_instance():
    inst = DockerLauncher(image="x:y")
    assert get_launcher(inst) is inst


def test_configured_launchers_not_resolvable_by_name():
    with pytest.raises(ValueError, match="needs configuration"):
        get_launcher("docker")
    with pytest.raises(ValueError, match="needs configuration"):
        get_launcher("sagemaker")


def test_unknown_launcher_name():
    with pytest.raises(ValueError, match="unknown launcher"):
        get_launcher("banana")


def test_all_launchers_satisfy_protocol():
    for inst in (SubprocessLauncher(), InProcessLauncher(), DockerLauncher(image="x:y")):
        assert isinstance(inst, ProbeLauncher)


# ---- DockerLauncher.build_run_command (pure) ------------------------------

def _probe(tmp_path: Path, instance_type="g5.xlarge") -> tuple[ProbeInput, Path]:
    probes = tmp_path / "probes"
    probes.mkdir(parents=True, exist_ok=True)
    probe = ProbeInput(
        spec_path=str(tmp_path / "spec.json"),
        config={"x": 1},
        instance_type=instance_type,
        subset_fraction=0.1,
        repetition=0,
        runtime={},
        output_dir=str(probes / "probe-0000.output"),
        result_path=str(probes / "probe-0000.result.json"),
    )
    input_path = Path(probe.result_path).with_suffix(".input.json")
    return probe, input_path


def test_docker_command_mounts_root_and_runs_probe(tmp_path):
    probe, input_path = _probe(tmp_path)
    launcher = DockerLauncher(image="myimg:latest")
    argv = launcher.build_run_command(probe, input_path, timeout=900)

    assert argv[:3] == ["docker", "run", "--rm"]
    # run_dir (tmp_path) is mounted at the same absolute path.
    assert "-v" in argv
    assert f"{tmp_path}:{tmp_path}" in argv
    # the probe command is unchanged from the subprocess contract.
    assert argv[-4:] == ["python", "-m", "mlprobe.probe", str(input_path)]
    assert "myimg:latest" in argv
    # timeout surfaces as the stop-timeout.
    assert "--stop-timeout" in argv and "900" in argv


def test_docker_gpu_auto_adds_gpus_for_gpu_target(tmp_path):
    probe, input_path = _probe(tmp_path, instance_type="g5.xlarge")   # A10G
    argv = DockerLauncher(image="i").build_run_command(probe, input_path, timeout=60)
    assert "--gpus" in argv and "all" in argv


def test_docker_gpu_auto_omits_gpus_for_cpu_target(tmp_path):
    probe, input_path = _probe(tmp_path, instance_type="c5.xlarge")   # CPU-only
    argv = DockerLauncher(image="i").build_run_command(probe, input_path, timeout=60)
    assert "--gpus" not in argv


def test_docker_gpu_explicit_false_and_string(tmp_path):
    probe, input_path = _probe(tmp_path, instance_type="g5.xlarge")
    off = DockerLauncher(image="i", gpus=False).build_run_command(probe, input_path, timeout=60)
    assert "--gpus" not in off
    dev = DockerLauncher(image="i", gpus="device=0").build_run_command(probe, input_path, timeout=60)
    assert "--gpus" in dev and "device=0" in dev


def test_docker_extra_mounts_env_workdir(tmp_path):
    probe, input_path = _probe(tmp_path)
    launcher = DockerLauncher(
        image="i", mounts=[("/host/repo", "/repo", "ro")],
        env={"HF_HOME": "/cache"}, workdir="/repo",
    )
    argv = launcher.build_run_command(probe, input_path, timeout=60)
    assert "/host/repo:/repo:ro" in argv
    assert "HF_HOME=/cache" in argv
    assert "-w" in argv and "/repo" in argv


def test_docker_per_target_image_override(tmp_path):
    probe, input_path = _probe(tmp_path, instance_type="p4d.24xlarge")
    launcher = DockerLauncher(
        image="default:cpu",
        image_for=lambda it: "gpu:img" if "p4d" in it else "default:cpu",
    )
    argv = launcher.build_run_command(probe, input_path, timeout=60)
    assert "gpu:img" in argv and "default:cpu" not in argv


def test_docker_missing_binary_returns_error_result(tmp_path):
    probe, _ = _probe(tmp_path)
    launcher = DockerLauncher(image="i", docker_bin="definitely-not-docker-xyz")
    result = launcher.launch(probe, timeout=10)
    assert result.error == "DockerNotFound"


# ---- real parity test (skipped without Docker / network) ------------------

def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def _bind_mount_visible(host_dir: Path) -> bool:
    """Whether the daemon can actually bind-mount ``host_dir`` (some sandboxed
    daemons can't see the caller's filesystem — e.g. host /tmp). The launcher
    relies on bind mounts, so if even this fails, parity can't be tested here."""
    marker = host_dir / "._mlprobe_mount_probe"
    marker.write_text("ok")
    try:
        out = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{host_dir}:{host_dir}",
             "python:3.12-slim", "cat", str(marker)],
            capture_output=True, text=True, timeout=120,
        )
        return out.returncode == 0 and "ok" in out.stdout
    except Exception:
        return False
    finally:
        marker.unlink(missing_ok=True)


PARITY_DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /repo
COPY pyproject.toml /repo/
COPY mlprobe /repo/mlprobe
COPY examples /repo/examples
RUN pip install --no-cache-dir -e .
ENV PYTHONPATH=/repo
"""


@pytest.fixture(scope="module")
def parity_image():
    if not _docker_available():
        pytest.skip("docker unavailable")
    tag = "mlprobe-test:parity"
    df = REPO_ROOT / "Dockerfile.parity-test"
    df.write_text(PARITY_DOCKERFILE)
    try:
        build = subprocess.run(
            ["docker", "build", "-f", str(df), "-t", tag, str(REPO_ROOT)],
            capture_output=True, text=True, timeout=600,
        )
        if build.returncode != 0:
            pytest.skip(f"image build failed (likely no network):\n{build.stderr[-800:]}")
        yield tag
    finally:
        df.unlink(missing_ok=True)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def _fake_probe(run_dir: Path) -> ProbeInput:
    from mlprobe import (
        DatasetSpec,
        EvalMetric,
        ModelSpec,
        ProbeConfig,
        TargetInstance,
    )

    spec = ModelSpec(
        name="parity",
        train_callable="examples.fake_trainable:train",
        evaluate_callable="examples.fake_trainable:evaluate",
        dataset=DatasetSpec(loader="examples.fake_trainable:load", total_size=50_000),
        eval_metrics=[EvalMetric(name="f1_macro", higher_is_better=True, primary=True),
                      EvalMetric(name="inference_latency_ms", higher_is_better=False)],
        targets=[TargetInstance(instance_type="c5.xlarge")],
        probe=ProbeConfig(subset_fractions=[0.1], timeout_seconds=120),
    )
    (run_dir / "probes").mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    return ProbeInput(
        spec_path=str(spec_path), config={"encoder": "small", "batch_size": 32, "epochs": 1},
        instance_type="c5.xlarge", subset_fraction=0.1, repetition=0, runtime={},
        output_dir=str(run_dir / "probes" / "probe-0000.output"),
        result_path=str(run_dir / "probes" / "probe-0000.result.json"),
    )


@pytest.fixture
def parity_workdir():
    """A run directory the Docker daemon can bind-mount. Placed under the repo
    (not pytest's /tmp tmp_path) because some sandboxed daemons can't see the
    caller's /tmp; skips if even repo-local mounts aren't visible."""
    import tempfile

    base = Path(tempfile.mkdtemp(prefix=".parity-", dir=REPO_ROOT))
    if not _bind_mount_visible(base):
        shutil.rmtree(base, ignore_errors=True)
        pytest.skip("docker daemon cannot bind-mount the host filesystem here")
    try:
        yield base
    finally:
        _force_rmtree(base)


def _force_rmtree(path: Path) -> None:
    """Remove a dir even if the container (running as root) left root-owned
    files our user can't delete — fall back to a throwaway root container."""
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{path}:/clean",
             "python:3.12-slim", "rm", "-rf", "/clean"],
            capture_output=True, timeout=120,
        )
        path.rmdir() if path.exists() else None


def test_subprocess_docker_parity(parity_image, parity_workdir):
    """Same probe, both launchers, identical deterministic outcome fields."""
    sub_probe = _fake_probe(parity_workdir / "sub")
    sub = SubprocessLauncher().launch(sub_probe, timeout=120)

    dock_probe = _fake_probe(parity_workdir / "dock")
    # The image is self-contained (mlprobe + examples baked in, PYTHONPATH=/repo);
    # the launcher's automatic run-dir mount is all that's needed.
    docker = DockerLauncher(image=parity_image).launch(dock_probe, timeout=300)

    assert sub.error is None, sub.traceback
    assert docker.error is None, docker.traceback
    # Timing/cost differ between hosts; the computed outcomes must not.
    assert sub.eval_metrics == docker.eval_metrics
    assert sub.train_metrics == docker.train_metrics
    assert sub.config == docker.config
