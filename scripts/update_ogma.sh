#!/usr/bin/env bash
# The one command that brings Ogma (standard-part physics + Yax, shared with Piccolo) into vendor/ogma, by git subtree.
# It is the only path that contacts the Ogma remote; builds, installs and activate never call it.
#
#   scripts/update_ogma.sh REF [REMOTE]      pull REF (branch, tag or commit) into vendor/ogma (--add the first time)
#   scripts/update_ogma.sh --check           print the recorded revision and whether vendor/ogma still matches it
#
# REMOTE defaults to $OGMA_REMOTE, else https://github.com/technolog-spectrumai/ogma (a local path works too).
# It needs a clean working tree, records the revision in vendor/OGMA_REVISION and runs Ogma's tests before
# committing. Make shared changes in Ogma first, then run this in Vegeta and Piccolo with the same REF.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PREFIX=vendor/ogma
REV=vendor/OGMA_REVISION
DEFAULT_REMOTE=https://github.com/technolog-spectrumai/ogma

if [[ "${1:-}" == "--check" ]]; then
    [[ -f $REV ]] || { echo "no $REV: Ogma is not vendored"; exit 1; }
    cat "$REV"
    split=$(git log -1 --grep="^git-subtree-dir: $PREFIX/*\$" --format=%B | sed -n 's/^git-subtree-split: //p')
    rec=$(sed -n 's/^revision: //p' "$REV")
    [[ "$split" == "$rec" ]] && echo "ok: the last subtree squash is $rec" || { echo "MISMATCH: last subtree squash $split, recorded $rec"; exit 1; }
    git diff --quiet HEAD -- "$PREFIX" || echo "note: $PREFIX has uncommitted local changes (make them in Ogma instead)"
    exit 0
fi

REF="${1:?usage: scripts/update_ogma.sh REF [REMOTE] | --check}"
REMOTE="${2:-${OGMA_REMOTE:-$DEFAULT_REMOTE}}"
git diff --quiet && git diff --cached --quiet || { echo "the working tree is not clean: commit or stash first"; exit 1; }

echo "fetching $REF from $REMOTE"
git fetch --no-tags "$REMOTE" "$REF"
SHA=$(git rev-parse FETCH_HEAD)
SHORT=$(git rev-parse --short FETCH_HEAD)

if [[ -n "$(git ls-files -- "$PREFIX" | head -1)" ]]; then
    git subtree merge --prefix="$PREFIX" --squash -m "Update vendored Ogma to $SHORT" "$SHA"
else
    git subtree add --prefix="$PREFIX" --squash -m "Vendor Ogma $SHORT with git subtree" "$SHA"
fi

if [[ "$(git config --bool commit.gpgsign || true)" == "true" ]] && ! git cat-file -p HEAD | grep -q '^gpgsig'; then
    merge=HEAD
    squash=$(git rev-parse "$merge^2" 2>/dev/null || true)
    if [[ -n "$squash" ]] && git log -1 --format=%B "$squash" | grep -q "^git-subtree-dir:"; then
        new_squash=$(git log -1 --format=%B "$squash" | git commit-tree -S $(git cat-file -p "$squash" | sed -n 's/^parent /-p /p' | tr '\n' ' ') "$squash^{tree}")
        new_merge=$(git log -1 --format=%B "$merge" | git commit-tree -S -p "$merge^1" -p "$new_squash" "$merge^{tree}")
        git reset -q --soft "$new_merge"
    fi
fi

printf 'revision: %s\nref: %s\nremote: %s\nupdated: %s\n' "$SHA" "$REF" "$REMOTE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$REV"
"$PREFIX/scripts/test.sh" -q -p no:cacheprovider
git add "$REV"
git commit -qm "Record vendored Ogma revision $SHORT"
echo "vendor/ogma is Ogma $SHA; now run the host smoke tests (./test.sh --smoke) and push"
