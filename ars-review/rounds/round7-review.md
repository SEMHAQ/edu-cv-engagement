# ARS Round 7 Review

**Paper**: Deep Learning for Student Engagement Detection: A Lightweight CNN Approach for Educational Environments  
**Journal Target**: Education Sciences (MDPI, APA format)  
**Date**: 2026-06-07  
**Previous Round**: Round 6 (Originality 82, Methodology 85, Evidence 88, Coherence 87, Writing 86)

---

## Dimension Scores (Round 7)

| Dimension | Weight | R6 Score | R7 Score | Δ | Weighted |
|---|---|---|---|---|---|
| Originality | 20% | 82 | 84 | +2 | 16.8 |
| Methodological Rigor | 25% | 85 | 91 | +6 | 22.75 |
| Evidence Sufficiency | 25% | 88 | 89 | +1 | 22.25 |
| Argument Coherence | 15% | 87 | 91 | +4 | 13.65 |
| Writing Quality | 15% | 86 | 89 | +3 | 13.35 |
| **Overall** | **100%** | **85.6** | **88.8** | **+3.2** | **88.8** |

---

## Detailed Assessment

### 1. Originality (Score: 84/100)

**Strengths:**
- The paper presents a coherent framework that integrates four components (CSAM, focal weighted loss, emotion-to-engagement mapping, deployment-oriented evaluation) into a unified system. While individual components are established, the integration is purposeful and well-justified.
- CSAM's three design choices for mobile architectures (r=4, single spatial branch, per-block insertion) are clearly differentiated from CBAM and SE-Net, with empirical validation showing 2.4--3.7% improvement over alternatives (Table 7).
- The emotion-to-engagement mapping is grounded in three complementary theoretical frameworks (flow theory, control-value theory, circumplex model), which elevates this beyond a simple relabeling exercise.
- DAiSEE validation (Table 9) provides independent confirmation that features generalize beyond emotion-to-engagement mapping.

**Weaknesses:**
- CSAM remains structurally similar to CBAM with mobile-specific parameter adjustments; the innovation is engineering optimization rather than a novel attention paradigm.
- The focal weighted loss combines existing components (focal loss, class weights, label smoothing, weighted sampling) without introducing a novel loss formulation.
- The emotion-to-engagement mapping, while theoretically grounded, produces proxy labels rather than ground-truth engagement annotations.
- No comparison with Transformer-based attention mechanisms (e.g., ViT variants or efficient attention modules), which limits the novelty claim for the attention contribution.

**Specific suggestions:**
1. Strengthen the "research gap" articulation in the Introduction to explicitly identify what prior work lacks (e.g., no prior work combines lightweight mobile attention with focal loss for engagement detection).
2. Add a brief comparison or at least a discussion of Transformer-based attention in Related Work to position CSAM against modern alternatives.
3. Frame the novelty more explicitly around the "deployment-aware engagement detection pipeline" rather than individual components.

---

### 2. Methodological Rigor (Score: 91/100)

**Strengths:**
- 5-fold cross-validation with mean±std is consistently applied across all experiments.
- 95% confidence intervals are reported for key comparisons (line 341), with explicit formula (x̄ ± 1.96 × SE).
- Two-stage transfer learning protocol is well-specified (Stage 1: 10 epochs lr=10⁻³, Stage 2: 40 epochs lr=10⁻⁴ with step decay).
- Comprehensive ablation studies isolate contributions: augmentation (+3.8%), transfer learning (+9.2%), CSAM (+1.3--5.0%), focal loss (+1.5--2.4% F1 on minority classes).
- Statistical significance reported with paired t-tests and Cohen's d effect sizes.
- Focal loss γ sensitivity analysis (Table 4) validates design choices.
- The training strategy confound is explicitly addressed by applying full pipeline to all baselines (Table 8).

**Weaknesses:**
- The CK+ cross-dataset evaluation does not use cross-validation; it evaluates FER2013-trained models on the CK+ test split, which is acceptable for a held-out evaluation but differs protocol-wise from the FER2013 experiments.
- No comparison of learning rate schedules (e.g., cosine annealing vs. step decay).
- The "Cohen's d > 2.0" claim for all pairwise comparisons is stated but not individually computed in a table.

**Specific suggestions:**
1. State the CK+ evaluation protocol more explicitly to distinguish it from the 5-fold CV protocol on FER2013.
2. Consider adding a brief note on why step LR was chosen over alternatives.

---

### 3. Evidence Sufficiency (Score: 89/100)

**Strengths:**
- Three datasets (FER2013, CK+, DAiSEE) provide multiple validation angles.
- Comprehensive ablation: augmentation (Table 1), class balancing (Table 2), focal loss γ (Table 3), transfer learning (Table 5), CSAM (Table 6), attention comparison (Table 7).
- Main results table (Table 8) includes both baseline-pipeline and full-pipeline comparisons for fairness.
- Computational efficiency comparison (Table 9) with params, FLOPs, FPS, and energy.
- Per-class analysis (Table 10) identifies Engaged (75.5% F1) vs. Confusion (56.4% F1) performance gap.
- Grad-CAM visualizations (Figure 4) provide qualitative evidence of attention patterns.
- Cross-dataset evaluation with honest discussion of limitations (114 test samples, ±8% CI).
- DAiSEE binary validation (74.5% accuracy, Table 9) with per-fold breakdown.

**Weaknesses:**
- CK+ per-class analysis is mentioned in text ("Engaged class transfers best (F1: 0.88), while Boredom transfers poorly (F1: 0.42)") but not presented in a table, making it harder to verify.
- The SOTA comparison table (Table 11) is limited; only 7 methods are compared, and some are from different tasks (pure FER vs. engagement detection).
- No statistical significance test for cross-dataset results.
- The DAiSEE F1 for Disengaged (0.33) is very low; this should be discussed more as it reveals a fundamental challenge with class-imbalanced engagement data.

**Specific suggestions:**
1. Add a per-class CK+ table to make the cross-dataset analysis more verifiable.
2. Expand SOTA comparison with more recent engagement detection methods.
3. Add a brief statistical note on cross-dataset result reliability given the small sample size.

---

### 4. Argument Coherence (Score: 91/100)

**Strengths:**
- The paper follows a clear narrative arc: problem → related work → method → experiments → discussion → conclusion.
- The progression FER2013 → CK+ → DAiSEE is logical: first establishing the framework, then testing generalization, then validating on engagement-specific data.
- The training strategy confound is proactively addressed (applying full pipeline to baselines), which strengthens the architectural advantage claim.
- Limitations section is comprehensive (8 limitations) and honest, covering mapping validity, label noise, static images, class boundaries, small CK+ sample, pre-cropped faces, training confound, and cultural bias.
- Consistency: all numerical claims in the abstract match the tables and text.

**Weaknesses:**
- The CSAM mechanism is explained three times: in the Methods section (lines 149-165), in the Results section (lines 253-298), and in the Discussion (lines 499-503). While some repetition is expected, the triple explanation could be consolidated.
- The "Novelty and Contribution" subsection (Section 4.3) lists four contributions that overlap significantly with the Introduction's contribution list.

**Specific suggestions:**
1. Reduce CSAM re-explanation in the Discussion to a brief summary that refers back to the Methods section.
2. In Section 4.3, focus on synthesizing contributions rather than re-listing them.

---

### 5. Writing Quality (Score: 89/100)

**Strengths:**
- APA format is generally well-followed throughout.
- The abstract is comprehensive and includes key quantitative results.
- Professional academic tone with appropriate hedging language.
- Citations consistently use \citep format.
- Tables and figures are well-formatted and informative.
- The limitations section demonstrates intellectual honesty and maturity.

**Weaknesses:**
- Bibliography entries use "and others" instead of "et al." (goodfellow2013challenges, dhebe2025engagement, manu2026video, russakovsky2015imagenet) — this produces "others" in the rendered APA output.
- The `manu2026video` entry has "Manu, others" as author which renders incorrectly.
- Some sentences are overly long (>50 words), particularly in the Discussion section.
- The contribution list in Section 4.3 uses bold formatting inconsistently (some contributions have bold labels, others don't).
- A few minor redundancies in phrasing across the Results and Discussion sections.

**Specific suggestions:**
1. Fix all "and others" in bib entries to use proper BibTeX "and Others" or restructure author lists.
2. Fix the `manu2026video` author entry.
3. Break a few of the longest sentences into two.

---

## Summary

The paper has improved substantially across all dimensions since Round 6. The most significant improvements are in Methodological Rigor (+6) and Argument Coherence (+4), driven by the addition of 95% confidence intervals, Cohen's d effect sizes, attention mechanism comparison table, and training protocol specification.

**Overall score: 88.8** — approaching but not yet reaching the 90 threshold.

**Remaining gaps to 90:**
1. **Originality (84)**: Structurally limited by the contribution type (engineering combination rather than conceptual breakthrough). Improvement requires reframing rather than new content.
2. **Evidence (89)**: Missing per-class CK+ table and expanded SOTA comparison.
3. **Writing (89)**: Bib entry fixes needed.

**Recommendation:** The paper is near-ready for submission to Education Sciences. The remaining improvements are: (a) strengthen originality framing, (b) add per-class CK+ table, (c) fix bib entries, and (d) minor writing polish.

---

## Priority Improvements to Reach ≥90

1. **[HIGH] Originality framing**: Add explicit research gap statement in Introduction; strengthen the "deployment-aware pipeline" novelty framing; add Transformer attention discussion.
2. **[HIGH] Evidence**: Add per-class CK+ cross-dataset table (Table); expand SOTA comparison.
3. **[MEDIUM] Writing**: Fix "and others" in 4 bib entries; fix manu2026video author.
4. **[LOW] Coherence**: Trim CSAM re-explanation in Discussion.
5. **[LOW] Methodology**: Note CK+ evaluation protocol distinction.
