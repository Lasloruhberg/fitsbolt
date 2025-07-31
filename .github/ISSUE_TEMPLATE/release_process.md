---
name: Release procedure
about: Make a new release
title: ''
labels: ''
assignees: ''

---
# Release
 
## What Needs to Be Done (chronologically)

- [ ] Create a new branch from `main` called `release` (e.g. `release-0.1.0`)
- [ ] Write changelog into `CHANGELOG.md`
- [ ] Minimize and update packages in `pyproject.toml` based on `environment.yml`
- [ ] Check unit tests -> Check all tests pass and that there are tests for all important features
- [ ] Check documentation -> Check presence of documentation for all new or changed user-facing features in README.md
- [ ] Change version number in `pyproject.toml` and __init__.py
- [ ] Create PR: `release` → `main`
- [ ] Test that you can locally `pip install -e .` the module
- [ ] Run Upload Python Package to testpypi workflow on release branch
- [ ] Check it was successful and try to install it in a blank conda env for sanity
- [ ] Request and run PR Review once successful 
- [ ] Merge `release` into `main`
- [ ] Create Release on GitHub from the last commit (the one reviewed in the PR) reviewed
- [ ] Trigger Upload Python Package to pypi workflow
 