#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Restore a data snapshot previously shipped with scripts/upload_data.sh.
# Fetches the snapshot tag, extracts the tree, reassembles any byte-sharded
# files, and verifies every file against MANIFEST.sha256. Files already
# present with a matching checksum are left untouched (safe to re-run).
#
# Usage:
#   bash scripts/fetch_data.sh <tag> [target-dir]
#
# Examples:
#   bash scripts/fetch_data.sh processed-v1            # restore under repo root
#   bash scripts/fetch_data.sh processed-v1 /tmp/x     # ...or into another dir
#
# Env overrides:
#   SLYTRADE_DATA_REMOTE  git remote to fetch from (default: `origin`'s URL)
#
# Requires: git, sha256sum
# Disk note: the snapshot is staged in a temp dir first, so you need ~2x the
#            snapshot size free during the restore.
# ---------------------------------------------------------------------------
set -euo pipefail

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
    echo "usage: bash scripts/fetch_data.sh <tag> [target-dir]" >&2
    exit 1
fi

command -v git >/dev/null 2>&1 || { echo "error: git not found" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum not found" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${2:-$ROOT}"
mkdir -p "$TARGET"
REMOTE="${SLYTRADE_DATA_REMOTE:-origin}"

echo "Fetching snapshot tag '$TAG' from $REMOTE ..."
git fetch --force --quiet "$REMOTE" "refs/tags/$TAG:refs/tags/$TAG"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Extracting snapshot ..."
git archive "$TAG" | tar -x -C "$STAGE"

MANIFEST="$STAGE/MANIFEST.sha256"
if [[ ! -f "$MANIFEST" ]]; then
    echo "error: snapshot '$TAG' has no MANIFEST.sha256 — was it created by upload_data.sh?" >&2
    exit 1
fi

n="$(wc -l < "$MANIFEST")"
echo "Restoring $n file(s) into $TARGET ..."

i=0
fail=0
while IFS=' ' read -r sum size rel partinfo; do
    i=$((i + 1))
    dst="$TARGET/$rel"

    # Fast path: already on disk with the right checksum.
    if [[ -f "$dst" ]] && [[ "$(sha256sum "$dst" | cut -d' ' -f1)" == "$sum" ]]; then
        echo "[$i/$n] unchanged   $rel"
        continue
    fi

    mkdir -p "$(dirname "$dst")"
    if [[ "${partinfo:-}" == parts=* ]]; then
        # Reassemble byte-shards (glob expansion is sorted => part00..partNN order)
        cat "$STAGE/$rel".part* > "$dst"
        rm -f "$STAGE/$rel".part*
    else
        cp "$STAGE/$rel" "$dst"
        rm -f "$STAGE/$rel"
    fi

    got="$(sha256sum "$dst" | cut -d' ' -f1)"
    if [[ "$got" != "$sum" ]]; then
        echo "[$i/$n] CHECKSUM FAIL  $rel" >&2
        fail=$((fail + 1))
    else
        echo "[$i/$n] restored    $rel"
    fi
done < "$MANIFEST"

if [[ "$fail" -gt 0 ]]; then
    echo "Finished with $fail error(s)." >&2
    exit 1
fi
echo "Done. $n file(s) verified under $TARGET"
