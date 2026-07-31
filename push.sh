#!/bin/sh
# Push using a PAT held outside the repo. The token is never committed.
# Store it once with: printf '%s' "<token>" > ~/.ccr-secrets/pat && chmod 600 ~/.ccr-secrets/pat
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=0 \
GIT_ASKPASS="$HOME/.ccr-secrets/askpass.sh" GIT_TERMINAL_PROMPT=0 \
GIT_SSL_CAINFO=/root/.ccr/ca-bundle.crt \
  git push https://github.com/SatyakiMandal/GSL.git "$BRANCH:$BRANCH"
