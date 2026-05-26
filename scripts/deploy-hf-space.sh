#!/usr/bin/env bash
#
# Deploy the backend/ folder as a Hugging Face Space.
#
# Usage:
#   scripts/deploy-hf-space.sh <hf-username> [space-name]
#
# Example:
#   scripts/deploy-hf-space.sh cyril-36 prepbuddy
#
# The first time you run this you'll need:
#   - The HF Space already created at huggingface.co/spaces/<user>/<name>
#     (Docker SDK, blank template)
#   - A write-scope HF access token from huggingface.co/settings/tokens
#     git will prompt for it as the password (your HF username is the user)
#
# After the first push, set the GROQ_API_KEY and FRONTEND_URL secrets on the
# Space's Settings page so the FastAPI service can boot.

set -euo pipefail

HF_USER="${1:?HF username required: scripts/deploy-hf-space.sh <user> [space]}"
SPACE_NAME="${2:-prepbuddy}"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"
WORK_DIR="$(mktemp -d -t prepbuddy-hf-XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "==> Cloning ${SPACE_URL} into ${WORK_DIR}"
git clone "${SPACE_URL}" "${WORK_DIR}"

cd "${WORK_DIR}"

echo "==> Syncing backend files into the Space repo"
# rsync with --delete-excluded so removed files in backend/ disappear from the
# Space too, but keep the Space's own .git/ intact.
rsync -a --delete \
    --exclude='.git/' \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='data/sessions.db*' \
    --exclude='data/llm_cache.db*' \
    --exclude='.cache/' \
    --exclude='.DS_Store' \
    --exclude='tests/' \
    --exclude='evaluation/' \
    "${BACKEND_DIR}/" ./

git add -A

if git diff --cached --quiet; then
    echo "==> No changes to deploy."
    exit 0
fi

# Build a short commit message from the source repo's HEAD.
SOURCE_SHA="$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)"
SOURCE_MSG="$(cd "${REPO_ROOT}" && git log -1 --pretty=%s)"
git commit -m "deploy from ${SOURCE_SHA}: ${SOURCE_MSG}"

echo "==> Pushing to ${SPACE_URL}"
git push origin main

echo
echo "Deployed."
echo "Space: ${SPACE_URL}"
echo "Logs:  ${SPACE_URL}/logs"
echo "Build progress: ${SPACE_URL}"
