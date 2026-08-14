# The Visual Computer Spatially Reliable Editing Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete English Springer Nature LaTeX manuscript package for *The Visual Computer* that presents footprint-aware spatially reliable semantic editing for animatable Gaussian avatars and maps every headline claim to frozen project evidence.

**Architecture:** The manuscript package is evidence-first. A deterministic asset-preparation script copies and hashes the frozen figures and tabular sources, while a static manuscript checker validates journal structure, citations, labels, assets, abstract length, keywords, required qualifiers, and author-input markers. The authored `main.tex` remains a single submission-ready source file, and separate Chinese handoff documents preserve claim provenance and required author actions.

**Tech Stack:** Springer Nature `sn-jnl` LaTeX class, BibTeX with numbered Springer references, Python 3 standard library, matplotlib for the method overview, existing CSV/JSON/PDF/PNG experiment assets, ARS academic-paper workflow.

---

## File Structure

### Files to create

- `paper/the_visual_computer/main.tex`: single-file English manuscript source, including all tables and declarations.
- `paper/the_visual_computer/references.bib`: independently verified bibliography.
- `paper/the_visual_computer/sn-jnl.cls`: unmodified official Springer Nature class.
- `paper/the_visual_computer/sn-mathphys-num.bst`: unmodified official numbered bibliography style.
- `paper/the_visual_computer/figures/method_overview.pdf`: vector method overview.
- `paper/the_visual_computer/figures/method_overview.png`: raster preview of the method overview.
- `paper/the_visual_computer/figures/hair_fixed_comparison.pdf`: frozen fixed-view Hair comparison.
- `paper/the_visual_computer/figures/shoes_fixed_comparison.pdf`: frozen fixed-view Shoes comparison.
- `paper/the_visual_computer/figures/leakage_retention_curve.pdf`: frozen leakage-retention curve.
- `paper/the_visual_computer/figures/per_part_iou.pdf`: frozen per-part spatial metric figure.
- `paper/the_visual_computer/figures/temporal_actionable_leakage.pdf`: supplementary leakage-variation curve.
- `paper/the_visual_computer/source_tables/evidence_manifest.json`: source paths, SHA256 hashes, row counts, and copied-asset hashes.
- `paper/the_visual_computer/source_tables/internal_main.csv`: five-subject aggregate source snapshot.
- `paper/the_visual_computer/source_tables/internal_matched_retention.csv`: five-subject matched-retention source snapshot.
- `paper/the_visual_computer/source_tables/internal_statistics.csv`: five-subject paired statistics source snapshot.
- `paper/the_visual_computer/source_tables/ablation_components.csv`: A0--A6 source snapshot.
- `paper/the_visual_computer/source_tables/ablation_micro.csv`: A5 micro-ablation source snapshot.
- `paper/the_visual_computer/source_tables/real_editing.csv`: coverage-constrained edit source snapshot.
- `paper/the_visual_computer/source_tables/external_comparisons.csv`: external paired-comparison source snapshot.
- `paper/the_visual_computer/source_tables/temporal_main.csv`: scoped temporal source snapshot.
- `paper/the_visual_computer/scripts/prepare_manuscript_assets.py`: deterministic evidence snapshot and asset-copy tool.
- `paper/the_visual_computer/scripts/make_method_overview.py`: publication figure generator.
- `paper/the_visual_computer/scripts/check_manuscript.py`: fail-closed static manuscript validator.
- `paper/the_visual_computer/README_编译与投稿说明.md`: compile, packaging, and submission instructions.
- `paper/the_visual_computer/论文证据与主张映射.md`: argument blueprint and exact claim-to-source mapping.
- `paper/the_visual_computer/投稿前待作者填写项.md`: authorship, funding, declarations, release URLs, biographies, and photo checklist.
- `paper/the_visual_computer/内部审稿报告.md`: ARS five-dimension review and resolved/unresolved findings.

### Existing evidence read without modification

- `configs/semantic/frozen_a5_main_method_v1.json`
- `exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/aggregate/`
- `exp/acceptdata/unified_a5_paper_ablation_20260723/aggregate/`
- `exp/acceptdata/five_subject_real_editing_matched_strength_20260723/aggregate/formal_coverage_constrained/`
- `exp/acceptdata/four_method_paper_evidence_20260813/`
- `docs/四方法外部对比定性统计与时序证据记录_20260813.md`
- `docs/当前空间低泄漏编辑创新与TheVisualComputer投稿判断_20260810.md`
- `/remote-home/ming/Paper/2312.00860.pdf`
- `/remote-home/ming/Paper/2312.09228.pdf`
- `/remote-home/ming/Paper/2408.09665.pdf`

### Existing project files that remain untouched

- Training, rendering, dataset, and evaluation code outside `paper/the_visual_computer/`.
- Frozen experiment output under `exp/acceptdata/`.
- User-authored dirty worktree files unrelated to the manuscript package.

---

### Task 1: Scaffold the Official Springer Manuscript Package

**Files:**
- Create: `paper/the_visual_computer/sn-jnl.cls`
- Create: `paper/the_visual_computer/sn-mathphys-num.bst`
- Create: `paper/the_visual_computer/README_编译与投稿说明.md`
- Create: `paper/the_visual_computer/投稿前待作者填写项.md`

- [ ] **Step 1: Download the official December 2024 Springer Nature template to a temporary directory**

Run:

```bash
template_dir=$(mktemp -d /tmp/tvc-springer-template.XXXXXX)
curl -L --fail --silent --show-error \
  https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12 \
  -o "$template_dir/springer-template.zip"
unzip -t "$template_dir/springer-template.zip"
```

Expected: `No errors detected in compressed data` and an archive containing `sn-article-template/sn-jnl.cls` plus `bst/sn-mathphys-num.bst`.

- [ ] **Step 2: Extract only the official class and bibliography style**

Run:

```bash
mkdir -p paper/the_visual_computer
unzip -j "$template_dir/springer-template.zip" \
  sn-article-template/sn-jnl.cls \
  sn-article-template/bst/sn-mathphys-num.bst \
  -d paper/the_visual_computer
```

Expected: the two files exist and preserve the archive bytes.

- [ ] **Step 3: Write the compile and submission guide**

Use `apply_patch` to create `README_编译与投稿说明.md` with:

```markdown
# The Visual Computer 论文编译与投稿说明

主文件为 `main.tex`，模板为 Springer Nature December 2024 `sn-jnl`。

## 编译

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 投稿边界

- The Visual Computer 使用 single-blind review，最终标题页需填写真实作者信息。
- 摘要必须为 150--250 词，关键词必须为 4--6 个。
- 主文使用方括号数字引用。
- `AUTHOR_INPUT_REQUIRED` 标记必须在投稿前全部清零。
- SG-GS 必须描述为 released-code controlled-input canonical adaptation。
- GG 的 CoreView 377 仅达到 40% retention，不进入共同 60% 主统计。
- 时序结果只能称为 frame-to-frame leakage variation。
```

- [ ] **Step 4: Write the author-input checklist**

Use `apply_patch` to create `投稿前待作者填写项.md` with checkboxes for author order, corresponding author, affiliations, email, ORCID, CRediT roles, funding/no-funding confirmation, competing interests, code URL, processed-data URL, biographies, photographs, and author approval.

- [ ] **Step 5: Verify the scaffold**

Run:

```bash
test -s paper/the_visual_computer/sn-jnl.cls
test -s paper/the_visual_computer/sn-mathphys-num.bst
rg -n 'AUTHOR_INPUT_REQUIRED|single-blind|150--250|4--6' \
  paper/the_visual_computer/*.md
```

Expected: exit code 0 and all required boundaries appear.

- [ ] **Step 6: Commit the scaffold**

```bash
git add paper/the_visual_computer/sn-jnl.cls \
  paper/the_visual_computer/sn-mathphys-num.bst \
  paper/the_visual_computer/README_编译与投稿说明.md \
  paper/the_visual_computer/投稿前待作者填写项.md
git commit -m "docs: scaffold The Visual Computer manuscript"
```

---

### Task 2: Build a Deterministic Evidence Snapshot

**Files:**
- Create: `paper/the_visual_computer/scripts/prepare_manuscript_assets.py`
- Create: `paper/the_visual_computer/source_tables/*.csv`
- Create: `paper/the_visual_computer/source_tables/evidence_manifest.json`
- Create: `paper/the_visual_computer/figures/*.pdf`

- [ ] **Step 1: Write the asset-preparation script**

Use `apply_patch` to create a script with these fixed mappings:

```python
TABLE_SOURCES = {
    "internal_main.csv": "exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/aggregate/main_table.csv",
    "internal_matched_retention.csv": "exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/aggregate/matched_retention_table.csv",
    "internal_statistics.csv": "exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/aggregate/paired_statistics.csv",
    "ablation_components.csv": "exp/acceptdata/unified_a5_paper_ablation_20260723/aggregate/component_table.csv",
    "ablation_micro.csv": "exp/acceptdata/unified_a5_paper_ablation_20260723/aggregate/a5_micro_ablation_table.csv",
    "real_editing.csv": "exp/acceptdata/five_subject_real_editing_matched_strength_20260723/aggregate/formal_coverage_constrained/paired_statistics.csv",
    "external_comparisons.csv": "exp/acceptdata/four_method_paper_evidence_20260813/significance/comparisons.csv",
    "temporal_main.csv": "exp/acceptdata/four_method_paper_evidence_20260813/temporal/main_table.csv",
}

FIGURE_SOURCES = {
    "hair_fixed_comparison.pdf": "exp/acceptdata/four_method_paper_evidence_20260813/qualitative/fixed_main/hair_three_subject_five_method.pdf",
    "shoes_fixed_comparison.pdf": "exp/acceptdata/four_method_paper_evidence_20260813/qualitative/fixed_main/shoes_three_subject_five_method.pdf",
    "leakage_retention_curve.pdf": "exp/acceptdata/four_method_paper_evidence_20260813/canary/a5_377_eval/leakage_retention_curve.pdf",
    "per_part_iou.pdf": "exp/acceptdata/four_method_paper_evidence_20260813/canary/a5_377_eval/per_part_iou.pdf",
    "temporal_actionable_leakage.pdf": "exp/acceptdata/four_method_paper_evidence_20260813/temporal/curves/actionable_leakage.pdf",
}
```

The script must resolve the repository root, reject missing or empty inputs, copy with `shutil.copy2`, count CSV records with `csv.DictReader`, compute SHA256 before and after copying, reject hash mismatches, and write sorted UTF-8 JSON.

- [ ] **Step 2: Run the script**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  paper/the_visual_computer/scripts/prepare_manuscript_assets.py
```

Expected: a manifest with eight table entries and five figure entries; every copied hash equals its source hash.

- [ ] **Step 3: Verify frozen evidence invariants**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python - <<'PY'
import csv, json
from pathlib import Path
root = Path('paper/the_visual_computer')
manifest = json.loads((root / 'source_tables/evidence_manifest.json').read_text())
assert len(manifest['tables']) == 8
assert len(manifest['figures']) == 5
with (root / 'source_tables/external_comparisons.csv').open() as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 12
assert {row['comparison_method'] for row in rows} == {'saga', 'gaussian_grouping', 'sggs'}
with (root / 'source_tables/temporal_main.csv').open() as handle:
    temporal = list(csv.DictReader(handle))
assert {row['method'] for row in temporal} == {'a5', 'saga', 'gaussian_grouping', 'sggs'}
print('evidence snapshot: passed')
PY
```

Expected: `evidence snapshot: passed`.

- [ ] **Step 4: Commit the evidence snapshot**

```bash
git add paper/the_visual_computer/scripts/prepare_manuscript_assets.py \
  paper/the_visual_computer/source_tables \
  paper/the_visual_computer/figures
git commit -m "docs: snapshot manuscript evidence and figures"
```

---

### Task 3: Verify and Assemble the Bibliography

**Files:**
- Create: `paper/the_visual_computer/references.bib`
- Create: `paper/the_visual_computer/论文证据与主张映射.md`

- [ ] **Step 1: Run PDF integrity preflight for the three supplied papers**

Run the ARS `pdf_read_preflight.py` once per PDF with a temporary pypdf environment. Require `PASS` and record the SHA256 values from the approved design specification.

- [ ] **Step 2: Build the minimum literature corpus**

Verify metadata from primary sources for these required groups:

```text
Representation: 3D Gaussian Splatting, 3DGS-Avatar
Animatable Gaussians: Animatable Gaussians, Human Gaussian Splats, SplattingAvatar, GoMAvatar
Semantic Gaussians: SAGA, Gaussian Grouping, SG-GS, GaussianEditor
2D supervision: Segment Anything, human parsing reference used by the project
Avatar/dataset foundations: SMPL, ZJU-MoCap
Evaluation/statistics: only sources actually needed to justify metrics or statistical procedures
```

For each entry, verify at least one DOI, official proceedings record, or exact arXiv identifier. Do not include references discovered only through an unverified secondary page.

- [ ] **Step 3: Write `references.bib`**

Use stable semantic keys such as:

```bibtex
kerbl2023gaussians
qian2024_3dgsavatar
cen2025saga
ye2024gaussiangrouping
zhao2024sggs
kirillov2023sam
loper2015smpl
peng2021zjumocap
```

Every entry must include title, author, year, publication venue or arXiv identifier, and DOI URL when available.

- [ ] **Step 4: Write the claim-evidence and argument blueprint**

Create `论文证据与主张映射.md` with a row for each abstract/contribution claim:

```markdown
| Claim ID | Manuscript wording boundary | Frozen source | Exact statistic | Allowed scope |
|---|---|---|---|---|
| C1 | reduces actionable leakage against Voting | internal matched-retention CSV | 30.39% | five ZJU-MoCap subjects |
| C2 | lower than SAGA, GG, and SG-GS adaptation | external comparisons CSV | 90.44%, 83.46%, 93.31% | frozen three-subject protocol; GG common subset |
| C3 | improves Macro mIoU and Boundary F1 externally | external comparisons CSV | corrected p < 0.05 | parser-target protocol |
| C4 | lower frame-to-frame leakage variation | temporal main/comparisons CSV | 83.33%--93.41% | leakage metric only, not perceptual flicker |
```

Also include the section outline, evidence assigned to every section, counterarguments, and limitation responses.

- [ ] **Step 5: Audit the bibliography shape**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python - <<'PY'
from pathlib import Path
import re
text = Path('paper/the_visual_computer/references.bib').read_text()
keys = re.findall(r'@[A-Za-z]+\{([^,]+),', text)
assert len(keys) >= 15
assert len(keys) == len(set(keys))
for required in ['kerbl2023gaussians', 'qian2024_3dgsavatar', 'cen2025saga',
                 'ye2024gaussiangrouping', 'zhao2024sggs']:
    assert required in keys
print(f'bibliography: {len(keys)} unique entries')
PY
```

Expected: at least 15 unique verified entries and all five core keys.

- [ ] **Step 6: Commit the bibliography and blueprint**

```bash
git add paper/the_visual_computer/references.bib \
  paper/the_visual_computer/论文证据与主张映射.md
git commit -m "docs: map manuscript claims and verified references"
```

---

### Task 4: Generate the Method Overview Figure

**Files:**
- Create: `paper/the_visual_computer/scripts/make_method_overview.py`
- Create: `paper/the_visual_computer/figures/method_overview.pdf`
- Create: `paper/the_visual_computer/figures/method_overview.png`

- [ ] **Step 1: Write the figure generator**

Implement a horizontal four-stage matplotlib diagram with these exact stages:

```text
Multi-view parser evidence
-> Persistent canonical Gaussian semantics
-> Footprint target / boundary / outer evidence
-> Calibrated target and support edit weights
-> Local recolor / removal / texture edit
```

Use a white background, black/dark-gray typography, restrained blue for semantic evidence, green for target support, amber for allowed adjacency, and red only for actionable leakage. Use arrows between stages, define all labels in English, and export both vector PDF and 300-dpi PNG.

- [ ] **Step 2: Run the generator**

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  paper/the_visual_computer/scripts/make_method_overview.py
```

Expected: both outputs exist and are non-empty.

- [ ] **Step 3: Verify dimensions and nonblank content**

```bash
/opt/miniconda3/envs/ictrl/bin/python - <<'PY'
from pathlib import Path
from PIL import Image, ImageStat
p = Path('paper/the_visual_computer/figures/method_overview.png')
img = Image.open(p).convert('RGB')
assert img.width >= 2400 and img.height >= 900
stat = ImageStat.Stat(img)
assert min(stat.var) > 10
assert Path('paper/the_visual_computer/figures/method_overview.pdf').stat().st_size > 10_000
print(img.size, 'method overview: passed')
PY
```

Expected: a wide figure at least 2400 by 900 pixels with nonzero channel variance.

- [ ] **Step 4: Inspect the PNG directly**

Open the generated PNG with the local image viewer and confirm that all text fits, arrows do not overlap boxes, and the target/support/leakage legend is legible at manuscript width.

- [ ] **Step 5: Commit the method figure**

```bash
git add paper/the_visual_computer/scripts/make_method_overview.py \
  paper/the_visual_computer/figures/method_overview.pdf \
  paper/the_visual_computer/figures/method_overview.png
git commit -m "docs: add manuscript method overview"
```

---

### Task 5: Draft the Complete Springer Nature Manuscript

**Files:**
- Create: `paper/the_visual_computer/main.tex`

- [ ] **Step 1: Read the ARS drafting roles for this phase**

Read the complete `structure_architect_agent.md`, `argument_builder_agent.md`, and `draft_writer_agent.md`. Apply the approved configuration record and the claim-evidence blueprint without broadening the paper to perceptual temporal stability.

- [ ] **Step 2: Write the title page, abstract, and keywords**

Use:

```latex
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
\title[Footprint-Aware Semantic Editing]{Footprint-Aware Reliable Semantic Editing for Animatable Gaussian Avatars}
\author*[1]{\fnm{AUTHOR} \sur{INPUT REQUIRED}}\email{AUTHOR_INPUT_REQUIRED@example.com}
\affil*[1]{\orgdiv{AUTHOR_INPUT_REQUIRED}, \orgname{AUTHOR_INPUT_REQUIRED},
\orgaddress{\city{AUTHOR_INPUT_REQUIRED}, \country{AUTHOR_INPUT_REQUIRED}}}
```

Write a 180--220 word unstructured abstract with the three external actionable-leakage reductions and no citations. Provide exactly five keywords.

- [ ] **Step 3: Draft Introduction and Related Work**

Introduction sequence:

```text
animatable Gaussian avatar capability
-> need for part-level editing
-> why primitive center semantics leak in screen space
-> why matched retention is required
-> proposed framework
-> evidence summary
-> three contributions
```

Related Work subsections:

```text
Animatable Human Avatars with 3D Gaussians
Semantic Representations and Grouping in 3DGS
Localized Editing and Reliability Evaluation
```

Do not use novelty phrases stronger than the verified corpus permits.

- [ ] **Step 4: Draft Method with equations**

Define the canonical Gaussian set, semantic posterior, projected footprint evidence, boundary/adjacent support, target and outer activations, calibrated edit weights, matched-retention strength, raw leakage, allowed-adjacent leakage, and actionable leakage. Every symbol must be defined at first use, and the method overview must be cited as Fig. 1.

- [ ] **Step 5: Draft Experimental Setup**

Separate the five-subject internal protocol from the three-subject shared-40k external protocol. State all subjects, cameras, frames, six compact parts, frozen method selection, parser-target reference, operations, retention levels, 20,000-bootstrap setup, exact sign-flip test, and Holm correction. Preserve the GG-377 and SG-GS qualifiers.

- [ ] **Step 6: Draft Results and embed the five main tables**

Embed numeric tables directly in `main.tex`. Report:

```text
Internal A5 vs Voting at 50% and 60% retention
External actionable leakage, Macro mIoU, Boundary F1, CI, and Holm p
A0--A6 component ablation
A5 footprint/outer-penalty micro-ablation
Coverage-constrained recolor/removal/texture results
```

Discuss Hair and Shoes figures separately. Explicitly report the Shoes target-response limitation rather than interpreting low leakage as a uniformly stronger edit.

- [ ] **Step 7: Draft scoped temporal evidence, limitations, conclusion, and declarations**

The temporal paragraph must contain the phrase `frame-to-frame leakage variation` and must not contain `flicker-free`, `temporally stable editing`, or `perceptual temporal consistency improvement`. Declarations use `AUTHOR_INPUT_REQUIRED` markers for facts only the authors can supply.

- [ ] **Step 8: Check word allocation and prose quality**

Run a LaTeX-aware word count if available; otherwise strip commands conservatively with the static checker. Target 8,000--9,000 words, vary paragraph length, remove throat-clearing openers, avoid em-dash overuse, and keep results descriptive before interpretation.

- [ ] **Step 9: Commit the complete first draft**

```bash
git add paper/the_visual_computer/main.tex
git commit -m "docs: draft The Visual Computer manuscript"
```

---

### Task 6: Add a Fail-Closed Manuscript Checker

**Files:**
- Create: `paper/the_visual_computer/scripts/check_manuscript.py`

- [ ] **Step 1: Write the static checker**

The script must:

```python
REQUIRED_SECTIONS = [
    "Introduction",
    "Related Work",
    "Method",
    "Experimental Setup",
    "Results",
    "Discussion and Limitations",
    "Conclusion",
]
FORBIDDEN_CLAIMS = [
    "flicker-free",
    "temporally stable editing",
    "official reproduction of SG-GS",
    "reliable editing for all body parts",
]
REQUIRED_QUALIFIERS = [
    "controlled-input",
    "40\\% retention",
    "frame-to-frame leakage variation",
    "parser",
]
```

It must parse abstract and keyword macros, count 150--250 abstract words and 4--6 keywords, collect citation keys from `\cite{}`, collect BibTeX keys, reject missing and uncited core references, collect figure paths and reject missing/empty files, reject duplicate labels, reject unresolved `??`, require all seven sections, require the four qualifier families, reject forbidden claims case-insensitively, and print a structured pass/fail summary.

- [ ] **Step 2: Run a deliberate failing check**

Run the checker against a temporary one-line `.tex` fixture without sections.

Expected: nonzero exit and errors for missing abstract, keywords, sections, citations, and qualifiers.

- [ ] **Step 3: Run the checker on `main.tex`**

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  paper/the_visual_computer/scripts/check_manuscript.py \
  paper/the_visual_computer/main.tex \
  paper/the_visual_computer/references.bib
```

Expected: `manuscript static checks: passed` with abstract word count, keyword count, citation count, reference count, figure count, and section count.

- [ ] **Step 4: Commit the checker**

```bash
git add paper/the_visual_computer/scripts/check_manuscript.py
git commit -m "test: validate manuscript claims and assets"
```

---

### Task 7: Run ARS Citation Audit and Internal Peer Review

**Files:**
- Create: `paper/the_visual_computer/内部审稿报告.md`
- Modify: `paper/the_visual_computer/main.tex`
- Modify: `paper/the_visual_computer/references.bib`
- Modify: `paper/the_visual_computer/论文证据与主张映射.md`

- [ ] **Step 1: Read the review and citation roles**

Read the complete `citation_compliance_agent.md`, `abstract_bilingual_agent.md`, and `peer_reviewer_agent.md`. Use English-only abstract per the approved configuration despite the bilingual agent's default capability.

- [ ] **Step 2: Audit every citation**

Check citation existence, author/year/title agreement, DOI/arXiv identifiers, citation-to-claim fit, unused entries, and missing foundational references. Mark any source that remains unverifiable rather than inventing metadata.

- [ ] **Step 3: Recompute headline statistics independently**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  tools/summarize_four_method_paper_evidence.py verify \
  --output-root exp/acceptdata/four_method_paper_evidence_20260813
```

Then independently read the snapshot CSVs and assert the three external actionable-leakage estimates and relative reductions match the manuscript to the displayed precision.

- [ ] **Step 4: Simulate the ARS five-dimension review**

Write `内部审稿报告.md` with scores and findings for novelty/importance, methodology, evidence/statistics, clarity/organization, and journal fit. Critical findings block final formatting. Record whether each noncritical finding is fixed or moved to Acknowledged Limitations.

- [ ] **Step 5: Apply one targeted revision round**

Revise only findings grounded in the review. Do not introduce new experiments, broaden temporal claims, or replace verified numbers. Re-run the checker and evidence integrity verifier.

- [ ] **Step 6: Commit the audited draft**

```bash
git add paper/the_visual_computer/main.tex \
  paper/the_visual_computer/references.bib \
  paper/the_visual_computer/论文证据与主张映射.md \
  paper/the_visual_computer/内部审稿报告.md
git commit -m "docs: audit and revise journal manuscript"
```

---

### Task 8: Compile or Record the Explicit Compilation Boundary

**Files:**
- Modify: `paper/the_visual_computer/README_编译与投稿说明.md`
- Optional generated output: `paper/the_visual_computer/main.pdf`

- [ ] **Step 1: Detect an available compiler**

Run:

```bash
command -v latexmk || command -v pdflatex || command -v tectonic
```

Expected: a compiler path, or an empty result that triggers the documented no-compiler branch.

- [ ] **Step 2: Compile when a compiler is available**

For `pdflatex`:

```bash
cd paper/the_visual_computer
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: `main.pdf`, no LaTeX errors, no undefined citations, and no undefined references.

- [ ] **Step 3: Use the explicit fallback when no compiler is available**

Do not claim compilation success. Add this exact status to the README:

```markdown
## 当前编译状态

本机未安装 LaTeX 编译器；已通过静态结构、引用、标签、图片和主张边界检查，但 PDF 编译尚未验证。请将本目录上传 Overleaf，编译器选择 pdfLaTeX，并按本文档执行 BibTeX 编译链。
```

- [ ] **Step 4: Inspect compiled PDF when available**

Run PDF preflight, render or inspect representative pages, and verify title/abstract, method figure, external table, qualitative figures, limitations, and reference pages for clipping or overlap.

- [ ] **Step 5: Run final package checks**

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  paper/the_visual_computer/scripts/prepare_manuscript_assets.py
/opt/miniconda3/envs/ictrl/bin/python \
  paper/the_visual_computer/scripts/check_manuscript.py \
  paper/the_visual_computer/main.tex \
  paper/the_visual_computer/references.bib
/opt/miniconda3/envs/ictrl/bin/python \
  tools/summarize_four_method_paper_evidence.py verify \
  --output-root exp/acceptdata/four_method_paper_evidence_20260813
git diff --check -- paper/the_visual_computer
```

Expected: all commands exit 0. If compilation was unavailable, the README must contain the explicit unverified status.

- [ ] **Step 6: Commit final formatting state**

```bash
git add paper/the_visual_computer
git commit -m "docs: finalize The Visual Computer submission draft"
```

---

## Final Review Checklist

- [ ] The title, abstract, contributions, conclusion, and cover material all describe spatial reliability rather than comprehensive temporal stability.
- [ ] The abstract contains only frozen metrics and distinguishes the three-subject external protocol from the five-subject internal study.
- [ ] The external table reports GG on the common 60% subset and carries the CoreView 377 40% note.
- [ ] SG-GS is consistently labeled as a released-code controlled-input canonical adaptation.
- [ ] Parser-derived targets are never described as human ground truth.
- [ ] Shoes target-response weakness is visible in Results or Limitations.
- [ ] Every figure and table has a self-contained caption.
- [ ] All `AUTHOR_INPUT_REQUIRED` markers are listed in the Chinese author checklist.
- [ ] Static checks and evidence integrity checks pass freshly.
- [ ] PDF compilation is either freshly verified or explicitly reported as unavailable.
