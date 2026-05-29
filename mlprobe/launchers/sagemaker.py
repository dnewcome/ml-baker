"""SageMaker training-job launcher — SCAFFOLD (issue #24).

This is the cloud extension of the Docker launcher: a SageMaker training job is
a BYOC (bring-your-own-container) Docker run on managed hardware. It conforms to
the same ``ProbeLauncher`` protocol so it slots into the runner exactly like the
others.

STATUS: scaffold only. The structure, config surface, and protocol conformance
are in place, but every method that talks to AWS raises ``NotImplementedError``.
It was written without an AWS account to test against, so the cloud paths are
deliberately *not* presented as working — wiring + verifying them against real
SageMaker is the remaining work. ``boto3`` / ``sagemaker`` are optional deps
behind the ``[sagemaker]`` extra and are imported lazily so importing mlprobe
never requires them.

Design notes for whoever finishes this (from the issue):
  - Build/push to ECR is the *caller's* job; this launcher just needs an
    ``image_uri`` that already exists in ECR.
  - The standard BYOC pattern passes hyperparameters as **env vars**, mapped via
    ``env_map`` (param name -> SM env var name). That differs from the probe
    binary's read-a-JSON-file contract, so a real implementation must decide:
    either (a) keep the probe-JSON contract and stage the input on S3 + have the
    container pull it, or (b) translate ``probe.config`` to env vars per
    ``env_map`` for containers that already expect that. (a) keeps parity with
    the other launchers; (b) matches existing BYOC scripts like
    brand-clustering's ``deploy.sh``.
  - Stream CloudWatch logs while the job runs; collect metrics afterward (from
    MLflow or parsed stdout).
  - Respect a concurrency limit across simultaneous jobs.
"""

from __future__ import annotations

from mlprobe.launchers.base import error_result
from mlprobe.probe import ProbeInput, ProbeResult


class SagemakerLauncher:
    """Launch probes as SageMaker training jobs. **Scaffold** — see module docstring.

    Parameters
    ----------
    image_uri : ECR image URI (caller pre-builds/pushes; this launcher does not).
    role_arn : SageMaker execution role ARN.
    instance_type : SageMaker instance type (e.g. ``"ml.g5.8xlarge"``).
    region : AWS region.
    env_map : optional ``{param_name: SM_ENV_VAR}`` for BYOC containers that take
        hyperparameters via env vars.
    max_concurrent_jobs : cap on simultaneous training jobs.
    input_mode : ``"s3_probe_json"`` (stage probe input on S3, keep file
        contract) or ``"env_vars"`` (translate config via ``env_map``).
    s3_staging_uri : where probe inputs/outputs are staged when ``input_mode`` is
        ``"s3_probe_json"``.
    """

    name = "sagemaker"

    def __init__(
        self,
        *,
        image_uri: str,
        role_arn: str,
        instance_type: str,
        region: str = "us-east-1",
        env_map: dict[str, str] | None = None,
        max_concurrent_jobs: int = 4,
        input_mode: str = "s3_probe_json",
        s3_staging_uri: str | None = None,
    ):
        self.image_uri = image_uri
        self.role_arn = role_arn
        self.instance_type = instance_type
        self.region = region
        self.env_map = dict(env_map or {})
        self.max_concurrent_jobs = max_concurrent_jobs
        self.input_mode = input_mode
        self.s3_staging_uri = s3_staging_uri

    # -- lazy client (optional dep) --

    def _client(self):
        """Return a boto3 SageMaker client. Lazy so importing mlprobe never
        requires boto3; raises a clear install hint when the extra is missing."""
        try:
            import boto3  # noqa: F401
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "SagemakerLauncher needs the AWS SDK; install with "
                "pip install 'mlprobe[sagemaker]'"
            ) from e
        raise NotImplementedError(
            "SagemakerLauncher is a scaffold (issue #24): boto3 client wiring is "
            "not implemented yet."
        )

    # -- protocol --

    def launch(self, probe: ProbeInput, *, timeout: int) -> ProbeResult:
        """Launch one probe as a SageMaker training job. NOT IMPLEMENTED.

        Returns a structured error rather than raising, so a runner that mixes
        launchers doesn't abort the whole sweep — the failure is recorded on the
        probe like any other. The real implementation would: stage the probe
        input (per ``input_mode``), create the training job, stream CloudWatch
        logs until terminal, then read back the result (S3 or MLflow)."""
        return error_result(
            probe,
            error="NotImplemented:SagemakerLauncher",
            traceback=(
                "SagemakerLauncher is a scaffold (issue #24). Implement "
                "_create_training_job / _stream_logs / _collect_result against "
                "the sagemaker SDK, then remove this stub."
            ),
        )

    # -- pieces a real implementation needs (all TODO) --

    def _create_training_job(self, probe: ProbeInput, input_path):  # pragma: no cover
        raise NotImplementedError("create SageMaker training job (boto3 create_training_job)")

    def _stream_logs(self, job_name: str):  # pragma: no cover
        raise NotImplementedError("stream CloudWatch logs until the job reaches a terminal state")

    def _collect_result(self, probe: ProbeInput, job_name: str) -> ProbeResult:  # pragma: no cover
        raise NotImplementedError("read the result back from S3 / MLflow into a ProbeResult")
