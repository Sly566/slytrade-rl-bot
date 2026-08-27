#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ship data files to GitHub as an orphan snapshot TAG (transport only).
#
# Why a tag and not release assets?  The receiving sandbox can only reach
# github.com / api.github.com / git smart-HTTP — the release-asset CDN
# (objects.githubusercontent.com, uploads.github.com) is blocked there.
# Git objects travel through github.com, so a tag is the reliable carrier.
#
# Guarantees:
#   * Nothing touches the `main` branch or its history — the snapshot is a
#     single orphan commit reachable only via the tag ref.
#   * Files >90MB are byte-sharded (GitHub rejects single files >100MB);
#     fetch_data.sh reassembles them transparently.
#   * MANIFEST.sha256 in the snapshot lets the receiver verify every byte.
#
# Usage:
#   bash scripts/upload_data.sh <tag> [dir ...]
#
# Examples:
#   bash scripts/upload_data.sh processed-v1 data/processed
#   bash scripts/upload_data.sh data-v1 data/raw data/processed
#
# Receiver side (any clone of this repo):
#   bash scripts/fetch_data.sh <tag>
#
# Env overrides:
#   SLYTRADE_DATA_REMOTE  git remote to push to (default: `origin`'s URL)
#   GIT_REF               full ref to push (default: refs/tags/<tag>)
#
# Requires: git, sha256sum, split
# Note: paths must not contain spaces or glob characters. The standard
#       data/ partition layout satisfies this.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: bash scripts/upload_data.sh <tag> [dir ...]" >&2
    exit 1
fi

TAG="$1"; shift
DIRS=("$@")
if [[ ${#DIRS[@]} -eq 0 ]]; then
    DIRS=(data)
fi

command -v git >/dev/null 2>&1 || { echo "error: git not found" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum not found" >&2; exit 1; }
command -v split >/dev/null 2>&1 || { echo "error: split not found" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE="${SLYTRADE_DATA_REMOTE:-$(git remote get-url origin)}"
MAX_BYTES=$((90 * 1024 * 1024))   # stay under GitHub's 100MB-per-file cap

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
SNAP="$STAGE/snap"
MANIFEST="$STAGE/MANIFEST.sha256"
mkdir -p "$SNAP"
: > "$MANIFEST"

hr_bytes() { numfmt --to=iec "$1" 2>/dev/null || echo "$1 B"; }

count=0
bytes=0
shards=0

for d in "${DIRS[@]}"; do
    d="${d#./}"
    d="${d%/}"
    if [[ ! -d "$d" ]]; then
        echo "skip (not a directory): $d" >&2
        continue
    fi
    while IFS= read -r -d '' f; do
        rel="${f#./}"
        case "$rel" in
            *" "*|*'*'*|*'?'*)
                echo "error: unsupported path: $rel" >&2
                exit 1
                ;;
        esac
        sum="$(sha256sum "$f" | cut -d' ' -f1)"
        size="$(stat -c%s "$f")"
        dst="$SNAP/$rel"
        mkdir -p "$(dirname "$dst")"
        if (( size > MAX_BYTES )); then
            # Byte-shard oversized files; receiver concatenates part00..partNN
            split -b "${MAX_BYTES}" -d "$f" "$dst.part"
            n="$(ls "$dst".part* | wc -l)"
            printf '%s %s %s parts=%s\n' "$sum" "$size" "$rel" "$n" >> "$MANIFEST"
            shards=$((shards + n))
            echo "  sharded: $rel -> $n parts (single files must stay <100MB on GitHub)"
        else
            cp "$f" "$dst"
            printf '%s %s %s\n' "$sum" "$size" "$rel" >> "$MANIFEST"
        fi
        count=$((count + 1))
        bytes=$((bytes + size))
    done < <(find "$d" -type f \( -name '*.parquet' -o -name '*.zip' -o -name '*.csv.gz' \) -print0 | sort -z)
done

if [[ "$count" -eq 0 ]]; then
    echo "No .parquet/.zip/.csv.gz files found under: ${DIRS[*]}" >&2
    exit 1
fi

cp "$MANIFEST" "$SNAP/MANIFEST.sha256"

echo "Staged $count file(s), $(hr_bytes "$bytes") total ($shards shard(s))."
echo "Pushing snapshot tag '$TAG' to $REMOTE ..."

# Fresh throwaway repo: one orphan commit, pushed ONLY as the tag ref.
git init -q "$SNAP"
git -C "$SNAP" add -A
git -C "$SNAP" \
    -c user.name="SlyTrade data transfer" \
    -c user.email="slytrade-data@users.noreply.github.com" \
    commit -qm "data snapshot: $TAG ($count files, $(hr_bytes "$bytes"))"
git -C "$SNAP" remote add origin "$REMOTE"
git -C "$SNAP" push --force origin "HEAD:refs/tags/$TAG"

echo
echo "Done. Snapshot '$TAG' is on GitHub (orphan commit, main untouched)."
echo "Receiver runs:  bash scripts/fetch_data.sh $TAG"
