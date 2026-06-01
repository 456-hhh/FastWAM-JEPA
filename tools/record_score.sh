#!/usr/bin/env bash
# record_score.sh — commits current repo state, records score to scores.jsonl
set -euo pipefail

REPO=""
SCORES=""
ITER=""
IDEA_ID=""
TITLE=""
STATUS=""
PRIMARY=""
METRICS=""
NOTES=""
IS_BEST="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)     REPO="$2";     shift 2 ;;
    --scores)   SCORES="$2";   shift 2 ;;
    --iter)     ITER="$2";     shift 2 ;;
    --idea-id)  IDEA_ID="$2";  shift 2 ;;
    --title)    TITLE="$2";    shift 2 ;;
    --status)   STATUS="$2";   shift 2 ;;
    --primary)  PRIMARY="$2";  shift 2 ;;
    --metrics)  METRICS="$2";  shift 2 ;;
    --notes)    NOTES="$2";    shift 2 ;;
    --is-best)  IS_BEST="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Git commit
cd "$REPO"
git config user.name optimizer 2>/dev/null || true
git config user.email opt@local 2>/dev/null || true
git add -A 2>/dev/null || true
COMMIT_MSG="iter-${ITER}: ${IDEA_ID} - ${TITLE}"
git commit -q -m "$COMMIT_MSG" --allow-empty
COMMIT_HASH=$(git rev-parse HEAD)

# Update _best tag if this is new best
if [[ "$IS_BEST" == "true" ]]; then
  git tag -f _best
  echo "[record_score] Updated _best tag to $COMMIT_HASH"
fi

# Create scores dir if needed
mkdir -p "$(dirname "$SCORES")"

# Timestamp
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Append JSON line
JSON_LINE="{\"iter\": \"${ITER}\", \"idea_id\": \"${IDEA_ID}\", \"title\": \"${TITLE}\", \"status\": \"${STATUS}\", \"primary\": ${PRIMARY}, \"metrics\": ${METRICS}, \"notes\": \"${NOTES}\", \"is_best\": ${IS_BEST}, \"commit\": \"${COMMIT_HASH}\", \"timestamp\": \"${TS}\"}"

echo "$JSON_LINE" >> "$SCORES"
echo "[record_score] Recorded iter=${ITER} primary=${PRIMARY} is_best=${IS_BEST} commit=${COMMIT_HASH}"
