# Docker launcher

Run each probe inside a container instead of a local subprocess. Same probe
contract (`python -m mlprobe.probe <input.json>`) — the launcher just adds a
bind mount and a `docker run` wrapper.

```python
import mlprobe
from mlprobe import DockerLauncher

launcher = DockerLauncher(image="mlprobe-cpu:latest")
results = mlprobe.run(spec, run_dir="probe_runs/exp1", launcher=launcher)
```

Scenarios take it too:

```python
scaling_with_n.run(spec, target="g5.xlarge",
                   launcher=DockerLauncher(image="mlprobe-gpu:latest"))
```

## The image must import your code

The probe imports your `train`/`evaluate`/`load` callables by dotted path, so
the image needs **mlprobe + your training package** importable. The reference
Dockerfiles here install mlprobe and put the repo on `PYTHONPATH`; adapt them to
your package. Build from the **repo root** so the build context includes
`mlprobe/` and your code:

```bash
docker build -f examples/docker/Dockerfile.cpu -t mlprobe-cpu:latest .
docker build -f examples/docker/Dockerfile.gpu -t mlprobe-gpu:latest .   # PyTorch + CUDA
```

## How paths work

The launcher bind-mounts the run directory at the **same absolute path** inside
the container, so the absolute paths baked into the probe's input JSON resolve
identically in and out of the container, and the result file the probe writes
lands back on the host. No path rewriting.

## Options

| Option | Purpose |
|---|---|
| `image` | container image (must import mlprobe + your code) |
| `gpus` | `"auto"` (default; `--gpus all` when the target has GPUs), `True`/`False`, or a string like `"device=0"` |
| `mounts` | extra `(host, container, mode)` binds — e.g. a model/HF cache, or your code dir |
| `env` | env vars passed with `-e` (e.g. `HF_HOME`) |
| `workdir` | container working directory |
| `image_for` | `instance_type -> image` for per-target images |

## GPU caveat

`--gpus` exercises the **host's** GPUs, not the target instance's. That makes
the Docker launcher great for build-time correctness ("does this even run in the
container?") but not for true target-hardware cost/perf measurement — that's
what the SageMaker launcher (issue #24) is for.
