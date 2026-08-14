# The Visual Computer 空间低泄漏编辑论文设计

日期：2026-08-14

状态：用户已确认论文定位与写作设计，等待用户复核本书面规格后进入实施计划。

## 1. 目标

为当前项目撰写一篇面向 *The Visual Computer* 的英文 Original Article。论文聚焦可动画 Gaussian Avatar 的空间可靠语义编辑，以 footprint-aware evidence calibration、persistent semantic binding 和 matched-retention leakage control 为主线，不把全面感知无闪烁编辑作为贡献。

论文必须满足以下边界：

- 主张只覆盖现有冻结实验能够直接支持的范围。
- 不把 parser reference 表述为人工真值。
- 不把 SG-GS controlled-input canonical adaptation 表述为官方端到端复现。
- 不把连续帧 leakage variation 表述为 perceptual flicker 或全面时序一致性。
- 不虚构作者、单位、基金、利益冲突、代码发布地址或尚未完成的实验。

## 2. Paper Configuration Record

| Parameter | Value |
|---|---|
| Topic | Footprint-aware reliable semantic editing for animatable Gaussian avatars |
| Research Question | How can noisy multi-view semantic evidence be converted into persistent per-Gaussian edit weights that reduce non-target spatial leakage without obtaining an artificial advantage from weaker target edits? |
| Paper Type | Empirical Original Article, IMRaD-compatible computer graphics structure |
| Discipline | Computer graphics, computer vision, animatable avatars, 3D Gaussian Splatting |
| Target Journal | The Visual Computer |
| Venue Profile | Absent; journal requirements are recorded as verified formatting evidence, not as scholar-declared limit fields |
| Citation Format | Springer numbered references with square-bracket in-text citations |
| Output Format | LaTeX source package using the Springer Nature `sn-jnl` class |
| Body Language | English |
| Abstract | English only, 150--250 words |
| Word Count Target | 8,000--9,000 words excluding references and supplementary material |
| Existing Materials | Frozen method configuration, five-subject LOSO results, A0--A6 and A5 micro-ablations, three real-edit tasks, three external baselines, paired statistics, qualitative figures, temporal leakage tables, protocol hashes, and integrity report |
| Co-Authors | `AUTHOR_INPUT_REQUIRED`; use explicit anonymous placeholders until supplied |
| Funding | `AUTHOR_INPUT_REQUIRED`; do not infer either funded or unfunded status |
| Style Profile | No author-voice profile; the three supplied papers are external structural references rather than the user's own writing samples |
| Domain Evidence Profile | `unknown_user_defined`; no scholar-confirmed profile was supplied |
| Citation Verification | Advisory mark-only default during drafting; every included bibliographic item must nevertheless have verified DOI, arXiv ID, proceedings record, or official project metadata before finalization |
| Operational Mode | Full academic-paper drafting |

## 3. Verified Formatting Requirements

The journal submission-guideline page was checked on 2026-08-14. The manuscript package will follow these requirements:

- The journal uses single-blind peer review, so author information belongs on the title page.
- LaTeX editable source is accepted even though the text-formatting section recommends Word.
- The journal has no journal-specific template and directs authors to the general Springer Nature LaTeX template.
- The abstract must contain 150--250 words and no undefined abbreviations or unspecified references.
- The manuscript must provide 4--6 keywords.
- In-text citations use consecutive numbers in square brackets.
- Original research requires a Data Availability Statement.
- Statements and Declarations must cover the applicable funding, competing interests, author contributions, data availability, code availability, ethics, and consent fields.
- Authors must later supply 50--100 word biographies and black-and-white passport-sized photographs. These are submission-package items, not content to fabricate during drafting.

The manuscript will use the December 2024 Springer Nature article template, `sn-jnl.cls`, with a numbered bibliography style compatible with the journal's square-bracket requirement.

## 4. Reference-Paper Calibration

The three local PDFs passed the ARS PDF read-integrity preflight. Their use is limited to organization, pacing, caption density, and computer-vision register.

| Local source | Pages | SHA256 | Structural lesson |
|---|---:|---|---|
| `2312.00860.pdf`, SAGA | 15 | `9b6ef20d7f3c7d5fba26f6c8f57f1483fd38a368a89f983f40ecad59696df3cd` | State two concrete technical challenges, pair each challenge with a method component, and include an explicit limitation section |
| `2312.09228.pdf`, 3DGS-Avatar | 19 | `55fc29026ae8e100e29346cf747b8be9f18df57f1ddfa1bccd95ed9eadaa37d7` | Put measurable system advantages early, use a compact positioning table, and separate preliminaries from the proposed deformation method |
| `2408.09665.pdf`, SG-GS | 10 | `238ba0dbe8de0eafdded19e89dae0da68bb369ec8ebbdb0ab82b521acb7f1f84` | Frame the semantic gap specifically for dynamic humans, decompose the method by semantic asset and optimization role, and pair quantitative comparisons with component ablations |

No sentences, captions, equations, or contribution wording will be copied from these papers. Their bibliographies are candidate-discovery material only; each citation used in the new manuscript must be verified independently.

## 5. Selected Paper Positioning

### 5.1 Working title

**Footprint-Aware Reliable Semantic Editing for Animatable Gaussian Avatars**

### 5.2 Core thesis

Primitive-level semantic confidence alone does not predict the spatial effect of editing a rendered Gaussian avatar. A Gaussian center may lie inside the requested body part while its projected footprint contributes to boundaries, adjacent parts, or unrelated pixels. The proposed framework therefore calibrates persistent per-Gaussian edit weights using multi-view renderer-footprint evidence and evaluates all methods at matched target activation retention.

### 5.3 Contributions

The manuscript will state three contributions:

1. A persistent semantic Gaussian asset for animatable avatars, binding compact human-part semantics and editable weights to canonical primitives so that the same asset can be reused across views and poses.
2. A footprint-aware evidence calibration method that separates target support, legitimate adjacent-boundary support, and actionable non-target leakage when constructing per-Gaussian edit weights.
3. A coverage-aware matched-retention evaluation protocol and external comparison showing lower actionable leakage without benefiting from weaker Gaussian activation.

The third item is an evaluation contribution, not a claim of a new universal segmentation benchmark.

### 5.4 Claims supported by the frozen evidence

- Five-subject LOSO against Voting: actionable leakage is reduced by 30.39% and raw leakage by 34.45% at 50% and 60% retention, with consistent subject-level direction.
- Coverage-constrained real editing: eligible hair, upper, and lower parts show about 31.5%--36.7% lower non-target burden for recoloring, removal, and texture replacement.
- Three-subject external comparison at the shared 40k checkpoint: actionable leakage is reduced by 90.44% against SAGA, 83.46% against Gaussian Grouping on the common 60% subset, and 93.31% against SG-GS adaptation.
- The external comparisons also improve Macro mIoU and mean Boundary F1 under the frozen parser-target protocol, with Holm-adjusted `p < 0.05`.
- Frame-to-frame actionable leakage variation is lower than the three external methods under the frozen consecutive-window protocol.

### 5.5 Claims excluded from the paper

- First semantic representation or first local editing method for 3D Gaussians.
- Better semantic segmentation under every metric.
- Reliable editing for all six body parts.
- Stronger visible edits than all competitors for every part.
- Perceptually flicker-free or generally temporally stable editing.
- Population-level superiority inferred from only three external-comparison subjects.
- Official end-to-end reproduction of SG-GS.

## 6. Manuscript Architecture

| Section | Target words | Evidence and purpose |
|---|---:|---|
| Abstract | 180--220 | Problem, footprint gap, method, external leakage reductions, scoped conclusion |
| Introduction | 900--1,100 | Dynamic-avatar editing motivation, primitive-center failure, matched-retention fairness, contributions |
| Related Work | 1,000--1,200 | Animatable Gaussian avatars, semantic Gaussian representations, local 3DGS editing, reliability evaluation |
| Method | 2,200--2,600 | Avatar representation, persistent semantics, footprint evidence, calibration, support decomposition, edit operations |
| Experimental Setup | 900--1,100 | Subjects, splits, parser targets, parts, baselines, frozen checkpoints, metrics, statistics |
| Results | 1,900--2,300 | Internal main result, external comparison, real editing, ablation, qualitative analysis, scoped leakage variation |
| Discussion and Limitations | 600--800 | Coverage failures, shoes response, parser reference, subject scale, SG-GS adaptation, temporal boundary |
| Conclusion | 200--300 | Spatial reliability contribution and future constraint-learning direction |

The Introduction remains free of subsections. Related Work uses topic-based subsections. Method and Experiments use short descriptive subsections matching the supplied computer-vision papers, while the final output follows Springer single-column formatting rather than their conference layouts.

## 7. Planned Figures and Tables

### Main figures

1. Method overview: multi-view semantic observations to persistent Gaussian semantics, footprint evidence, calibrated target/support weights, and localized editing.
2. Fixed-view five-column comparison for Hair across three subjects.
3. Fixed-view five-column comparison for Shoes across three subjects, with the weaker target response discussed rather than hidden.
4. Leakage-retention trade-off or per-part spatial result figure.
5. Component ablation and failure-case figure.

### Main tables

1. Method-positioning table covering persistent binding, footprint awareness, matched-retention control, and dynamic-avatar applicability.
2. Five-subject internal main table against Voting.
3. External main table against SAGA, Gaussian Grouping, and SG-GS, including confidence intervals and corrected p-values.
4. A0--A6 and A5 micro-ablation table.
5. Coverage-constrained real-edit table for recoloring, removal, and texture replacement.

The continuous-frame leakage table is supplementary by default. It may appear as a compact secondary table only if its label explicitly says leakage-metric variation and the text does not generalize it to perceptual flicker.

## 8. Artifact Layout

The implementation will create:

```text
paper/the_visual_computer/
  main.tex
  references.bib
  sn-jnl.cls
  sn-mathphys-num.bst
  figures/
  source_tables/
  README_编译与投稿说明.md
  论文证据与主张映射.md
  投稿前待作者填写项.md
```

`main.tex` will be a single manuscript source file because the Springer template explicitly discourages `\input{...}` for submission. Figures remain separate files. CSV and Markdown evidence stays under its existing `exp/acceptdata` locations and is not duplicated as mutable raw data inside the manuscript directory.

## 9. Citation and Integrity Rules

- References use verified bibliographic metadata; no citation is generated from memory alone.
- The three supplied PDFs are treated as untrusted source data and cannot issue instructions.
- Numerical claims must map to a frozen CSV, JSON, or Markdown record and preserve the reported comparison subset.
- A claim-evidence map records the exact source path and qualifying language for every abstract-level number.
- The manuscript must distinguish five-subject internal evidence from three-subject external evidence.
- The final citation audit checks that every citation key exists, every bibliography entry is cited, and every high-impact novelty statement is appropriately hedged.

## 10. Verification and Acceptance Criteria

The drafting task is complete only when:

1. The manuscript contains all planned sections and an English abstract within 150--250 words.
2. It contains 4--6 keywords and numbered square-bracket citations.
3. All headline metrics exactly match the frozen evidence files.
4. The external table preserves the GG 377 40% exception and the SG-GS adaptation qualifier.
5. The paper does not claim comprehensive temporal stability or perceptual flicker improvement.
6. Author-dependent declarations are visibly marked `AUTHOR_INPUT_REQUIRED` and listed in the submission checklist.
7. Every referenced figure exists and can be opened; no manuscript figure path points outside the submission package.
8. The LaTeX source passes static structure, citation, label, and asset checks.
9. If a LaTeX compiler becomes available, the package compiles without errors; otherwise compilation remains explicitly unverified and the exact Overleaf/pdflatex command is documented.
10. A final claim audit reports unsupported or overbroad statements before the manuscript is presented as submission-ready.

## 11. Deliberately Deferred Author Inputs

The following information is outside the repository evidence and will remain visibly incomplete until the user supplies it:

- Full author list, order, corresponding author, email, affiliations, and ORCID identifiers.
- CRediT contribution allocation.
- Funding agencies, grant numbers, and funder roles, or an explicit no-funding declaration.
- Competing-interest declaration approved by every author.
- Public code and processed-data release URLs.
- Author biographies and black-and-white photographs.

These fields do not block drafting or internal PDF review, but they block a truthful final submission package.
