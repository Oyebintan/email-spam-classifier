# ECC (installed component)

This project vendors a curated, stack-relevant subset of [affaan-m/ECC](https://github.com/affaan-m/ECC)
(v2.0.0), a Claude Code agent/skill/rules framework. An earlier commit vendored the
full framework (734 files, ~7MB); it was trimmed down because Claude Code registers
every installed agent's and skill's name/description into context automatically each
session (visible as "New agent types/skills are now available" system messages) —
so unused entries cost real context tokens on every session, not just disk space.

- `.claude/agents/` — 9 agents relevant to a Python/Flask/scikit-learn stack:
  planner, code-reviewer, python-reviewer, security-reviewer, tdd-guide,
  mle-reviewer, build-error-resolver, doc-updater, performance-optimizer.
- `.claude/skills/` — 13 skills: git-workflow, error-handling,
  documentation-lookup, security-review, security-scan, verification-loop,
  database-migrations, api-design, deployment-patterns, python-patterns,
  python-testing, mle-workflow, tdd-workflow.
- `.claude/rules/ecc/` — common, python.

Not installed: the other ~265 skills, ~58 agents, ~20 rule packs, `commands/`
(legacy slash-command shims — superseded by skills, no value without them), and
`hooks/` (auto-executing scripts — deliberately excluded).

Installed manually (file copy only, per ECC's documented manual-install path) — no ECC
installer script, npm dependency, or hook automation was run or added.

Update by re-copying only the directories above from a fresh checkout of upstream, or
add more if a specific need comes up (e.g. add a rule pack only when the project
actually starts using that language/framework — don't install "just in case").

## License

ECC is MIT licensed:

```
MIT License

Copyright (c) 2026 Affaan Mustafa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
