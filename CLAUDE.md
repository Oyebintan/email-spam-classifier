# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project overview

A hybrid feature-selection pipeline (Chi-Square filter + L1-regularized logistic
regression) feeding a deep neural network for email spam/ham classification, plus
a Flask backend for serving inference (see `backend/`, `Procfile`, `convert_model.py`).

## ECC install

`.claude/` carries a project-level install of [ECC](https://github.com/affaan-m/ECC)
v2.0.0, trimmed to a curated, stack-relevant subset (9 agents, 13 skills, 2 rule
packs — no hooks, no legacy commands). See `.claude/ECC-NOTICE.md` for the full
list and why it's trimmed rather than the full ~734-file bundle.
