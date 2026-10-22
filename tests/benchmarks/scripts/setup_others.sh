#!/usr/bin/env bash
# Clone the two Python ports at the commits used for results/results.json.
# Usage: bash scripts/setup_others.sh <dir>
set -euo pipefail
dir="${1:?usage: setup_others.sh <dir>}"
mkdir -p "$dir"
clone() {
  git clone --quiet "$1" "$dir/$2"
  git -C "$dir/$2" checkout --quiet "$3"
}
clone https://github.com/NiRuff/decontx-python niruff cd86f0b1ab3808541e04acf2ff4d5dd53baf5a66
clone https://github.com/omicverse/py-decontx omicverse 4c97b3261351e0a2074b3dc947f177c0d39ec394
echo "Cloned into $dir/niruff and $dir/omicverse"
