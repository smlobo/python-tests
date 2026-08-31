#!/usr/bin/env python3
"""Clone a GitHub repository, detect projects, and build them."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import urllib.error
import urllib.request


@dataclass(frozen=True)
class Project:
    """Detected project rooted at a directory with a known build marker."""

    kind: str
    path: Path


@dataclass(frozen=True)
class CommandResult:
    """Captured result of a shell command."""

    command: list[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    """Raised when a command exits unsuccessfully."""

    def __init__(self, result: CommandResult):
        super().__init__(self._build_message(result))
        self.result = result

    @staticmethod
    def _build_message(result: CommandResult) -> str:
        return (
            f"command failed with exit code {result.returncode}: {' '.join(result.command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


@dataclass(frozen=True)
class CloneRetryDecision:
    """LLM recommendation for bounded clone retry behavior."""

    action: str
    reason: str
    corrected_repo_url: str | None = None


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LOCAL_LLM_BASE_URL = "http://127.0.0.1:8080"
# DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemma-4-E4B-it-UD-Q4_K_XL.gguf")


PROJECT_MARKERS: tuple[tuple[str, str], ...] = (
    ("package.json", "node"),
    ("pyproject.toml", "python"),
    ("requirements.txt", "python"),
    ("go.mod", "go"),
    ("pom.xml", "maven"),
    ("build.gradle", "gradle"),
    ("build.gradle.kts", "gradle"),
)


def run_command(command: list[str], cwd: Path) -> CommandResult:
    """Run a subprocess and capture stdout/stderr."""
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    result = CommandResult(
        command=command,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise CommandError(result)
    return result


def _truncate_for_prompt(text: str, limit: int = 4000) -> str:
    """Keep LLM input bounded while preserving the tail of failures."""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head}\n...\n{tail}"


def llm_available() -> bool:
    """Return True when the required API configuration is present."""
    base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if base_url.startswith(DEFAULT_LOCAL_LLM_BASE_URL):
        return True
    return bool(os.getenv("OPENAI_API_KEY"))


def is_supported_github_repo_url(repo_url: str) -> bool:
    """Return True for common GitHub SSH/HTTPS repository URL forms."""
    candidate = repo_url.strip()
    if candidate.startswith("git@github.com:") and len(candidate) > len("git@github.com:"):
        return True
    if candidate.startswith("https://github.com/") and len(candidate) > len("https://github.com/"):
        return True
    return False


def build_llm_endpoint(base_url: str) -> str:
    """Normalize a base URL into a responses endpoint URL."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/responses"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/responses"
    return f"{trimmed}/v1/responses"


def build_models_endpoint(base_url: str) -> str:
    """Normalize a base URL into a models listing endpoint URL."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/models"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/models"
    return f"{trimmed}/v1/models"


def extract_response_text(response_body: str) -> str:
    """Extract text from common OpenAI-compatible response shapes."""
    parsed = json.loads(response_body)

    output_text = parsed.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = parsed.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "".join(chunks)

    raise RuntimeError(f"unexpected LLM response format: {response_body}")


def extract_json_object(text: str) -> str:
    """Extract a JSON object from plain text or fenced Markdown."""
    candidate = text.strip()

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise RuntimeError(f"no JSON object found in model output: {text}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                json_text = candidate[start : index + 1]
                json.loads(json_text)
                return json_text

    raise RuntimeError(f"no complete JSON object found in model output: {text}")


def probe_local_llm(base_url: str, *, model: str) -> None:
    """Confirm the local LLM endpoint is reachable and appears usable."""
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    models_url = build_models_endpoint(base_url)
    models_request = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(models_request, timeout=15) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"local LLM probe failed at {models_url}: HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"local LLM probe failed at {models_url}: connection error: {exc.reason}") from exc

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"local LLM probe returned invalid JSON from {models_url}: {response_body}") from exc

    models = parsed.get("data")
    if not isinstance(models, list):
        raise RuntimeError(f"local LLM probe returned unexpected model payload from {models_url}: {response_body}")

    available_models: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            available_models.append(model_id)

    print(f"Local LLM endpoint reachable at {models_url}")
    if available_models:
        print(f"Local LLM models: {', '.join(available_models)}")
    if available_models and model not in available_models:
        print(
            f"Requested model '{model}' not listed by local endpoint; continuing anyway.",
            file=sys.stderr,
        )


def interpret_clone_failure(
    repo_url: str,
    result: CommandResult,
    *,
    model: str,
) -> CloneRetryDecision:
    """Ask an OpenAI-compatible model whether to retry a clone safely."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")
    url = build_llm_endpoint(base_url)
    prompt = {
        "repo_url": repo_url,
        "command": result.command,
        "cwd": str(result.cwd),
        "returncode": result.returncode,
        "stdout": _truncate_for_prompt(result.stdout),
        "stderr": _truncate_for_prompt(result.stderr),
        "allowed_actions": ["retry_same", "retry_shallow", "retry_corrected_url", "abort"],
        "instructions": (
            "Return strict JSON with keys action, reason, and corrected_repo_url. "
            "Pick retry_same when the clone likely failed because of a transient network issue. "
            "Pick retry_shallow when a shallow clone might reduce transfer cost or avoid a flaky fetch. "
            "Pick retry_corrected_url only when the repository URL appears to contain a typo and you can infer a corrected GitHub URL. "
            "When action is retry_corrected_url, set corrected_repo_url to the corrected GitHub SSH or HTTPS repository URL. "
            "Pick abort for auth, permissions, or anything unlikely to succeed on retry."
        ),
    }
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a build agent supervisor. "
                            "Return only JSON. Never suggest commands outside the allowed actions. "
                            "Only suggest corrected_repo_url values for GitHub repositories."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(prompt)}],
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"connection error: {exc.reason}") from exc

    try:
        output_text = extract_response_text(response_body)
        decision = json.loads(extract_json_object(output_text))
        action = str(decision["action"]).strip()
        reason = str(decision["reason"]).strip()
        corrected_repo_url = decision.get("corrected_repo_url")
    except (RuntimeError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unexpected LLM response format: {response_body}") from exc

    if action not in {"retry_same", "retry_shallow", "retry_corrected_url", "abort"}:
        raise RuntimeError(f"LLM returned unsupported action: {action}")
    if action == "retry_corrected_url":
        if not isinstance(corrected_repo_url, str) or not is_supported_github_repo_url(corrected_repo_url):
            raise RuntimeError(f"LLM returned invalid corrected_repo_url: {corrected_repo_url}")
        return CloneRetryDecision(
            action=action,
            reason=reason,
            corrected_repo_url=corrected_repo_url.strip(),
        )
    return CloneRetryDecision(action=action, reason=reason, corrected_repo_url=None)


def clone_repository(
    repo_url: str,
    workdir: Path,
    *,
    use_llm: bool,
    llm_model: str,
) -> Path:
    """Clone repo_url into a temp directory under workdir with optional LLM retry."""
    active_repo_url = repo_url
    target_dir = Path(
        tempfile.mkdtemp(prefix="repo-build-agent-", dir=str(workdir.resolve()))
    )
    print(f"Cloning {active_repo_url} into {target_dir}")

    clone_attempts = [
        ["git", "clone", active_repo_url, str(target_dir)],
    ]

    attempt = 0
    while attempt < len(clone_attempts):
        command = clone_attempts[attempt]
        attempt += 1
        try:
            run_command(command, cwd=workdir)
            return target_dir
        except CommandError as exc:
            if not use_llm:
                raise

            decision = interpret_clone_failure(active_repo_url, exc.result, model=llm_model)
            print(f"LLM clone decision: {decision.action} ({decision.reason})")

            if decision.action == "retry_same" and attempt == len(clone_attempts):
                shutil.rmtree(target_dir, ignore_errors=True)
                clone_attempts.append(["git", "clone", active_repo_url, str(target_dir)])
                continue
            if decision.action == "retry_shallow" and attempt == len(clone_attempts):
                shutil.rmtree(target_dir, ignore_errors=True)
                clone_attempts.append(
                    ["git", "clone", "--depth", "1", active_repo_url, str(target_dir)]
                )
                continue
            if decision.action == "retry_corrected_url" and attempt == len(clone_attempts):
                shutil.rmtree(target_dir, ignore_errors=True)
                active_repo_url = decision.corrected_repo_url or active_repo_url
                print(f"Retrying with corrected repo URL: {active_repo_url}")
                clone_attempts.append(["git", "clone", active_repo_url, str(target_dir)])
                continue
            raise

    raise RuntimeError("clone attempts exhausted")


def detect_projects(repo_path: Path) -> list[Project]:
    """Find project roots based on common build files."""
    seen: set[Path] = set()
    projects: list[Project] = []

    for marker, kind in PROJECT_MARKERS:
        for found in repo_path.rglob(marker):
            root = found.parent
            if root in seen:
                continue
            seen.add(root)
            projects.append(Project(kind=kind, path=root))

    projects.sort(key=lambda project: (str(project.path), project.kind))
    return projects


def command_exists(name: str) -> bool:
    """Return True if a command is available on PATH."""
    return shutil.which(name) is not None


def build_project(project: Project) -> None:
    """Run a standard build for one detected project."""
    print(f"\nBuilding [{project.kind}] {project.path}")

    if project.kind == "node":
        if command_exists("npm"):
            run_command(["npm", "install"], cwd=project.path)
            run_command(["npm", "run", "build"], cwd=project.path)
            return
        raise RuntimeError("npm is not installed")

    if project.kind == "python":
        if (project.path / "pyproject.toml").exists():
            run_command([sys.executable, "-m", "pip", "install", "-e", "."], cwd=project.path)
            return
        if (project.path / "requirements.txt").exists():
            run_command(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=project.path,
            )
            return
        raise RuntimeError("unsupported python project layout")

    if project.kind == "go":
        if command_exists("go"):
            run_command(["go", "build", "./..."], cwd=project.path)
            return
        raise RuntimeError("go is not installed")

    if project.kind == "maven":
        if command_exists("mvn"):
            run_command(["mvn", "clean", "package", "-DskipTests"], cwd=project.path)
            return
        raise RuntimeError("mvn is not installed")

    if project.kind == "gradle":
        if (project.path / "gradlew").exists():
            run_command(["chmod", "+x", "gradlew"], cwd=project.path)
            run_command(["./gradlew", "build", "-x", "test"], cwd=project.path)
            return
        if command_exists("gradle"):
            run_command(["gradle", "build", "-x", "test"], cwd=project.path)
            return
        raise RuntimeError("gradle is not installed and no gradlew wrapper was found")

    raise RuntimeError(f"unsupported project kind: {project.kind}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Clone a GitHub repository and build detected projects."
    )
    parser.add_argument("repo_url", help="GitHub repository URL to clone")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Directory used to create the temporary clone",
    )
    parser.add_argument(
        "--use-llm-clone-retry",
        action="store_true",
        help="Use an OpenAI-compatible model to interpret clone failures and allow one safe retry",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_OPENAI_MODEL,
        help="Model used for clone failure interpretation",
    )
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        help="Use a local OpenAI-compatible LLM at http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--test-local-llm",
        action="store_true",
        help="Probe the local LLM endpoint before clone retry logic runs",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.use_local_llm and "OPENAI_BASE_URL" not in os.environ:
            os.environ["OPENAI_BASE_URL"] = DEFAULT_LOCAL_LLM_BASE_URL
        if args.use_local_llm or args.test_local_llm:
            probe_local_llm(
                os.getenv("OPENAI_BASE_URL", DEFAULT_LOCAL_LLM_BASE_URL),
                model=args.llm_model,
            )
        use_llm = args.use_llm_clone_retry and llm_available()
        if args.use_llm_clone_retry and not use_llm:
            print(
                "LLM clone retry requested but no usable LLM endpoint is configured; continuing without LLM.",
                file=sys.stderr,
            )

        repo_path = clone_repository(
            args.repo_url,
            workdir,
            use_llm=use_llm,
            llm_model=args.llm_model,
        )
        projects = detect_projects(repo_path)
        print(f"Detected {len(projects)} project(s)")
        for project in projects:
            print(f"[{project.kind}] {project.path}")

        if not projects:
            print("No supported projects found.")
            return 0

        failures = 0
        for project in projects:
            try:
                build_project(project)
            except (RuntimeError, CommandError) as exc:
                failures += 1
                print(f"Build failed for {project.path}: {exc}", file=sys.stderr)

        if failures:
            print(f"\nCompleted with {failures} failed project build(s).", file=sys.stderr)
            return 1

        print("\nAll detected projects built successfully.")
        return 0
    except (RuntimeError, CommandError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
