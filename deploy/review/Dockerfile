FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
RUN pip install --no-cache-dir 'torch>=2.6,<3' 'torchvision>=0.21,<1' \
      --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
RUN python -c 'import subprocess,tomllib; subprocess.check_call(["pip","install","--no-cache-dir",*tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]])'
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

USER 1000:1000
EXPOSE 8768
CMD ["ai-spine-review", "--port", "8768", "--host", "0.0.0.0"]
