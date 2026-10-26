# CPU image for the test suite and the viewer, for machines without Python 3.12.
#   docker build -t latent-comm . && docker run --rm latent-comm
# Runs the same check.sh as a plain venv would (bash setup.sh && bash check.sh).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md setup.sh check.sh ./
COPY src ./src
COPY tests ./tests
COPY latent/data ./latent/data
COPY latent/results ./latent/results
RUN bash setup.sh
CMD ["bash", "check.sh"]
