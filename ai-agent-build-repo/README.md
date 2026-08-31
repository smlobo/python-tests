# ai-agent-build-repo

Simple starter for an AI-agent-style workflow that:

* clones a GitHub repository
* scans it for common project types
* runs standard build commands for each project
* can ask an LLM to interpret clone failures and choose one safe retry action

This version keeps deterministic execution in Python code and uses the LLM only for bounded interpretation:

* normal code does the shell execution
* the LLM reads `git clone` stdout and stderr on failure
* the LLM may choose one of four actions: `retry_same`, `retry_shallow`, `retry_corrected_url`, or `abort`

## Run

```bash
cd ai-agent-build-repo
./build-repo-agent.py https://github.com/example/example-repo.git
```

Optional output directory:

```bash
./build-repo-agent.py https://github.com/example/example-repo.git --workdir /tmp/agent-builds
```

Enable LLM-based clone retry:

```bash
export OPENAI_API_KEY=your_key_here
./build-repo-agent.py \
  https://github.com/example/example-repo.git \
  --use-llm-clone-retry
```

Optional OpenAI-compatible settings:

* `OPENAI_API_KEY`
* `OPENAI_BASE_URL`
* `OPENAI_MODEL`

Use a local LLM server at `http://127.0.0.1:8080`:

```bash
./build-repo-agent.py \
  https://github.com/example/example-repo.git \
  --use-llm-clone-retry \
  --use-local-llm
```

The local mode now probes the endpoint automatically before any clone-retry logic runs. It assumes an OpenAI-compatible API shape and does not require `OPENAI_API_KEY` unless your local server enforces auth.

Run the probe by itself:

```bash
./build-repo-agent.py \
  https://github.com/example/example-repo.git \
  --use-local-llm \
  --test-local-llm
```

## Supported project markers

* `package.json` -> Node.js
* `pyproject.toml` -> Python
* `requirements.txt` -> Python
* `go.mod` -> Go
* `pom.xml` -> Maven
* `build.gradle` / `build.gradle.kts` -> Gradle

## Notes

* The script skips nested duplicate detections for the same directory.
* Build commands are conservative and do not run tests by default.
* Required tools must already be installed locally, such as `git`, `npm`, `python3`, `mvn`, or `gradle`.
* The LLM layer is limited to clone-failure interpretation only; command execution remains hardcoded and allowlisted in Python.
* `--use-local-llm` sets `OPENAI_BASE_URL` to `http://127.0.0.1:8080` unless you already set a different base URL, and it automatically probes `/v1/models`.
* `--test-local-llm` is still available when you want to run the probe without relying on local mode for clone-retry behavior.
* Corrected URL retries are limited to GitHub SSH or HTTPS repository URLs.
