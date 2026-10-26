---
name: release
description: Cut a new mcpy-gcmc release to PyPI and GitHub.
disable-model-invocation: true
---

# Release mcpy-gcmc

The package is `mcpy-gcmc` on PyPI (the name `mcpy` belongs to another project).
`.github/workflows/release.yml` does the build and upload when a `v*` tag is pushed.

## Steps

1. On `main`, with a clean tree, run the `verify-gcmc` skill; CI on `main` is green.
2. Set `version` in `pyproject.toml` to the release number (remove `.dev0`, for example `1.6.0`). Commit it.
3. Give the user these commands to run themselves (Claude cannot push release tags here):
   ```
   ! git push origin main
   ! git tag v1.6.0 && git push origin v1.6.0
   ```
   The workflow checks that the tag matches `pyproject.toml`, uploads to PyPI by trusted publishing, and creates the GitHub Release (Zenodo archives it).
4. Write the release notes by hand from the commit bodies since the last tag (`git log v<last>..HEAD`).
   Group them as: fixes that change results, features, other fixes, removed APIs, docs.
   Describe what changed. Leave out "by @user in #PR" lines and review names.
   The workflow posts auto-generated notes; the user pastes your text into the GitHub release editor to replace them.
5. Bump `pyproject.toml` to the next `.dev0` (for example `1.7.0.dev0`) and commit.

Done when: `pip install mcpy-gcmc==<version>` works in a fresh env, the GitHub Release has the hand-written notes, and `main` is on the next `.dev0`.

Leave `CHANGELOG.md` to the maintainer.
