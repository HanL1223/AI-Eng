# =====================================================================
# docker-compose.yml -- Multi-Container Orchestration
# =====================================================================
# Week 6, Step 4 of 6
#
# WHAT IS THIS FILE?
# -------------------
# docker-compose.yml defines one or more containers ("services") and
# how they connect. It replaces long docker run commands with a
# declarative configuration file.
#
# Instead of:
#   docker run -p 8000:8000 --env-file .env -v ./docs:/app/docs \
#     -v ./logs:/app/logs --name sttm-api sttm-rag-api
#
# You write this file and run:
#   docker compose up
#
# Docker Compose reads this file and creates all the containers,
# networks, and volumes automatically.
#
# WHY USE COMPOSE FOR A SINGLE SERVICE?
# ----------------------------------------
# Even with one service, docker-compose.yml is valuable because:
#   1. Documents all configuration in one file (ports, volumes, env)
#   2. One command to start/stop everything (docker compose up/down)
#   3. Easy to add services later (Redis cache, monitoring, etc.)
#   4. Reproducible -- no forgotten command-line flags
#
# COMMANDS:
#   docker compose up          Start all services (foreground)
#   docker compose up -d       Start all services (background/detached)
#   docker compose down        Stop and remove all containers
#   docker compose logs        View logs from all services
#   docker compose logs -f     Follow logs (like tail -f)
#   docker compose build       Rebuild images (after code changes)
#   docker compose up --build  Rebuild and start in one command
#   docker compose ps          List running services
#
# dbt ANALOGY:
# docker-compose.yml is like profiles.yml + dbt_project.yml combined.
# It defines:
#   - WHAT runs (services = dbt models)
#   - WHERE it connects (ports/networks = database connections)
#   - WHAT data it uses (volumes = sources/seeds)
#   - HOW it is configured (environment = vars)
# =====================================================================


# ─── VERSION ───
# The version field is optional in Docker Compose v2+.
# Including it documents which Compose specification version you target.
# Modern Docker Compose (v2) ignores this field but it helps humans.


# ─── SERVICES ───
# Each service becomes a Docker container. The service name ("api")
# is also the hostname on the Docker network, so other services
# can reach it at http://api:8000.
services:

  api:
    # ── BUILD CONFIGURATION ──
    # Tells Docker Compose how to build the image.
    #
    # context: .     Use the current directory as the build context
    #                (the set of files Docker can access during build).
    # dockerfile:    Which Dockerfile to use (default: Dockerfile).
    build:
      context: .
      dockerfile: Dockerfile

    # ── IMAGE NAME ──
    # Name the built image so you can reference it without rebuilding.
    # Without this, Docker Compose generates a name like "ai-eng-api".
    image: sttm-rag-api:latest

    # ── CONTAINER NAME ──
    # By default, Docker Compose names containers like "ai-eng-api-1".
    # container_name gives it a fixed, readable name.
    container_name: sttm-rag-api

    # ── PORT MAPPING ──
    # Maps HOST_PORT:CONTAINER_PORT.
    #
    # "8000:8000" means:
    #   - The container listens on port 8000 (set in Dockerfile CMD)
    #   - Port 8000 on YOUR MACHINE forwards to the container
    #   - You access the API at http://localhost:8000
    #
    # If port 8000 is already in use (e.g., another service), change
    # the HOST port: "8080:8000" -> access at http://localhost:8080
    ports:
      - "8000:8000"

    # ── ENVIRONMENT VARIABLES ──
    # Pass secrets and configuration WITHOUT baking them into the image.
    #
    # env_file reads key=value pairs from a file, exactly like your
    # Python .env loading in rag.py and api_server.py.
    #
    # SECURITY: The .env file is in .dockerignore, so it is NOT copied
    # into the image. It is only read at container runtime.
    #
    # You can also set variables directly:
    #   environment:
    #     - ANTHROPIC_API_KEY=sk-ant-...
    # But env_file is safer (keeps secrets out of docker-compose.yml,
    # which you might commit to Git).
    env_file:
      - .env

    # ── VOLUMES ──
    # Mount host directories into the container. Changes in either
    # direction are immediately visible.
    #
    # ./docs:/app/docs
    #   Your STTM Excel files on the host -> /app/docs in the container.
    #   The container reads these files during startup (via load_documents).
    #   You can add/remove files on the host and restart the container
    #   to pick up changes.
    #
    # ./logs:/app/logs
    #   Query logs persist to the host filesystem. Without this volume,
    #   logs would be lost when the container stops (containers are ephemeral).
    #
    # GOTCHA: Volume paths are relative to docker-compose.yml location.
    # Use ./ prefix for relative paths (same directory as this file).
    volumes:
      - ./docs:/app/docs
      - ./logs:/app/logs

    # ── RESTART POLICY ──
    # What happens when the container crashes?
    #
    # "no":            Never restart (default)
    # "always":        Always restart (even after manual stop)
    # "on-failure":    Restart only on non-zero exit codes
    # "unless-stopped": Restart unless manually stopped
    #
    # "unless-stopped" is the best for development:
    #   - Auto-restarts after crashes
    #   - Stays stopped when you manually stop it
    #   - Survives host reboots (if Docker is configured to start at boot)
    restart: unless-stopped

    # ── HEALTH CHECK ──
    # Override or supplement the HEALTHCHECK in the Dockerfile.
    # docker compose ps shows the health status.
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
      # start_period: Grace period for the RAG pipeline to load.
      # During this time, health check failures do not count.
      # Your pipeline needs ~10-30 seconds to load documents and
      # build the vector store, depending on corpus size.

    # ── LOGGING ──
    # Configure Docker's log driver. The json-file driver writes
    # container logs to JSON files on the host.
    #
    # max-size: Maximum size of each log file before rotation.
    # max-file: Number of rotated files to keep.
    #
    # Without these limits, a busy container can fill your disk
    # with log files. 10MB x 3 files = 30MB max log storage.
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"


# =====================================================================
# EXTENDING LATER
# =====================================================================
# When you add more services (Phase 2+), they go under `services:`.
# Example: Adding a Redis cache for conversation memory:
#
#   services:
#     api:
#       ... (everything above) ...
#       depends_on:
#         - redis
#
#     redis:
#       image: redis:7-alpine
#       ports:
#         - "6379:6379"
#       volumes:
#         - redis_data:/data
#
#   volumes:
#     redis_data:
#
# depends_on ensures Redis starts BEFORE the API container.
# The API can then connect to Redis at redis:6379 (Docker networking
# automatically resolves the service name "redis" to its container IP).

