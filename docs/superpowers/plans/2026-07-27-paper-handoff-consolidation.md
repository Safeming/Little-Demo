# Paper Handoff Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the paper innovation handoff with all completed five-subject experiments, corrected statistical analysis, current claim boundaries, and the shortest remaining submission path.

**Architecture:** Preserve the July 18 document as historical context, add an authoritative status notice near the top, and append a dated consolidated update. Every quantitative claim is sourced from a generated CSV/JSON artifact; outdated fixed-strength ratios and mixed-protocol results remain explicitly historical.

**Tech Stack:** Markdown, Git, existing experiment CSV/JSON artifacts, shell verification with `rg`, `sed`, `awk`, and `git diff --check`.

---

### Task 1: Freeze the evidence inventory

**Files:**
- Read: `exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/aggregate/*`
- Read: `exp/acceptdata/unified_a5_paper_ablation_20260723/aggregate/*`
- Read: `exp/acceptdata/five_subject_real_editing_paper_20260723/aggregate/*`
- Read: `exp/acceptdata/five_subject_real_editing_matched_strength_20260723/aggregate/*`
- Read: `exp/acceptdata/five_subject_semantic_temporal_stability_20260724/aggregate/formal_matched_retention/*`

- [ ] **Step 1: Verify every formal artifact directory exists**

Run `test -f` for the LOSO main table, ablation component table, matched-strength aggregate table, and temporal formal table.

- [ ] **Step 2: Record only subject-equal means and subject-level bootstrap intervals**

Use the generated aggregate and paired-statistics tables as the sole source for final claims.

### Task 2: Update the handoff document

**Files:**
- Modify: `docs/当前项目论文创新点与快速投稿补强交接_20260718.md`

- [ ] **Step 1: Add an authoritative update notice near the document header**

State that the appended 2026-07-27 section supersedes earlier status judgments when they conflict.

- [ ] **Step 2: Append completed experiment records**

Add dated sections for five-subject LOSO, A5/A6 freeze, A0-A6 ablation, real editing, matched strength, temporal evaluation, and corrected pooled/coverage-constrained statistics.

- [ ] **Step 3: Append updated innovation analysis and claim boundaries**

Separate supported spatial leakage-control claims from unsupported universal coverage and temporal-stability claims.

- [ ] **Step 4: Append the remaining P0/P1/P2 submission route**

Prioritize pooled coverage-constrained real-editing tables, temporal diagnostic separation, external baselines, paper figures, complexity, and final method freeze.

### Task 3: Verify the consolidated document

**Files:**
- Verify: `docs/当前项目论文创新点与快速投稿补强交接_20260718.md`

- [ ] **Step 1: Verify all referenced result paths exist**

Run a shell loop over the formal directories and require zero missing paths.

- [ ] **Step 2: Scan for placeholders and contradictory current-status language**

Run `rg` for `TBD`, `TODO`, and obsolete claims such as “只有一个正式主体” outside the explicitly historical sections.

- [ ] **Step 3: Check Markdown patch integrity**

Run `git diff --check` and inspect the new headings and tables with `sed`.

### Task 4: Commit the record

**Files:**
- Add: `docs/当前项目论文创新点与快速投稿补强交接_20260718.md`
- Add: `docs/superpowers/plans/2026-07-27-paper-handoff-consolidation.md`

- [ ] **Step 1: Stage only the requested handoff and this execution plan**

Run `git add` with the two exact paths.

- [ ] **Step 2: Inspect the staged diff and commit in Chinese**

Commit message: `文档：补充五主体论文实验与投稿分析`.
