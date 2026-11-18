#!/usr/bin/env bash
# Clone verl and lmms-engine at the upstream commits we build on and apply our patches (patches/*.patch).
#   bash scripts/setup_third_party.sh [target dir, default <repo>/third_party]
# Then install both in your environment: pip install -e third_party/verl; pip install -e third_party/lmms-engine
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-$REPO/third_party}"
mkdir -p "$DST"
fetch() {  # name url commit patch
  if [ ! -d "$DST/$1/.git" ]; then git clone "$2" "$DST/$1"; fi
  if git -C "$DST/$1" apply --reverse --check "$REPO/patches/$4" 2>/dev/null; then
    echo "$1: patches/$4 is already applied"; return
  fi
  git -C "$DST/$1" checkout --detach "$3"
  git -C "$DST/$1" apply "$REPO/patches/$4"
  echo "$1: $3 + patches/$4"
}
fetch verl https://github.com/volcengine/verl 1876b06d0a3e4e71e06230be10af14492ca8a75b verl.patch                  # v0.9.1
fetch lmms-engine https://github.com/EvolvingLMMs-Lab/lmms-engine c127b0041fc229878989474cd53ab1286ab2d5a7 lmms-engine.patch
