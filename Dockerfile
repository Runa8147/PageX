# ---------------------------------------------------------------------------
# PageX headless CLI image - `pagex:cpu`
#
#   docker build -t pagex:cpu .
#   docker run --rm -v ~/docs:/docs -v pagex-data:/data pagex:cpu ingest /docs/x.pdf
#
# PAGEX_HOME=/data makes /data the single home of ALL state (pagex.toml,
# SQLite db, artifacts, models, logs), so a named volume keeps everything
# between runs and can be snapshotted on its own:
#
#   docker run --rm -v pagex-data:/data pagex:cpu models download
#   docker run --rm -v ~/pack:/pack -v pagex-data:/data pagex:cpu \
#       models install-pack /pack/pagex-models-pack.zip
#
# GGUF model weights are NEVER baked into the image (ADR-002) - see the
# model commands above.
#
# Build note: `pip install "/app[llm]"` compiles llama-cpp-python from
# source inside the image (build-essential + cmake below). That is correct
# but slow (~5-15 min). Faster alternative: preinstall a prebuilt CPU wheel
# and drop the compiler toolchain from the apt line:
#
#   RUN pip install --prefer-binary llama-cpp-python \
#       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
#
# Purging build-essential after the pip layer is possible but requires
# re-marking libstdc++6 as manual first (llama.cpp links against it);
# kept simple here - see docs/packaging.md.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# libglib2.0-0 / libgomp1: OpenCV + OpenMP runtime (llama.cpp links libgomp).
# build-essential + cmake: compile the `llm` extra (llama-cpp-python).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgomp1 \
      build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# /app is both the build context and the runtime cwd: paths.bundle_dir()
# falls back to Path.cwd() for an installed (non-frozen) package, which makes
# /app/samples the read-only demo corpus of `pagex demo`.
WORKDIR /app

COPY . /app

# `llm` extra (pyproject.toml) = llama-cpp-python. Without it the image would
# boot but ingest/search/ask could not run any model.
RUN pip install --no-cache-dir "/app[llm]"

# Fail the build early if a runtime dependency did not land in the image.
RUN python -c "import fitz, cv2, numpy, psutil, huggingface_hub, llama_cpp, pagex"

# All state lives under /data (see header).
ENV PAGEX_HOME=/data
VOLUME /data

ENTRYPOINT ["pagex"]
