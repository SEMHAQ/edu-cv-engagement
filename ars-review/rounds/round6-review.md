# ARS Review — Round 6

**Paper**: Deep Learning for Student Engagement Detection: A Lightweight CNN Approach for Educational Environments  
**Journal Target**: Education Sciences (MDPI, APA format)  
**Date**: 2026-06-07  

---

## Dimension Scores

| Dimension | Weight | Score | Weighted |
|---|---|---|---|
| Originality | 20% | 78 | 15.6 |
| Methodological Rigor | 25% | 88 | 22.0 |
| Evidence Sufficiency | 25% | 84 | 21.0 |
| Argument Coherence | 15% | 80 | 12.0 |
| Writing Quality | 15% | 86 | 12.9 |
| **Overall** | **100%** | | **83.5** |

---

## 1. Originality (Score: 78/100)

### Strengths
- The emotion-to-engagement mapping grounded in three complementary theoretical frameworks (flow theory, control-value theory, circumplex model) is well-executed and provides principled justification rather than ad hoc relabeling.
- The joint optimization framework (focal weighted loss + weighted sampling + label smoothing) addresses class imbalance and hard-example mining simultaneously.
- The deployment-oriented evaluation (params, FLOPs, energy, inference time) is a practical contribution for an Education Sciences audience.

### Weaknesses
- CSAM is architecturally very close to CBAM (channel then spatial attention); the modifications (r=4, single spatial branch, per-block insertion) are engineering choices rather than fundamental innovations. The novelty claim could be more modest.
- The emotion-to-engagement mapping, while theoretically grounded, is fundamentally a label remapping on existing datasets rather than learning engagement representations from engagement-labeled data.
- The ablation of focal loss parameters (gamma=0.5, 1.0, 1.5, 2.0) is thorough but the gamma=1.0 finding is not surprising given prior focal loss literature.

### Recommendations
- Frame CSAM explicitly as "an adaptation of the CBAM paradigm for mobile inverted residual blocks" rather than implying greater novelty.
- Add a sentence acknowledging that the gamma=1.0 result is consistent with prior work while noting that confirming it in this specific context has value.

---

## 2. Methodological Rigor (Score: 88/100)

### Strengths
- 5-fold stratified cross-validation with mean±std is properly conducted and reported.
- Five complementary ablation experiments (augmentation, class balancing, gamma, transfer learning, CSAM) systematically isolate each contribution.
- The full-pipeline baseline comparison (Table 2, "full*" rows) addresses the training strategy confound identified in prior rounds — this is excellent methodological practice.
- Paired t-tests are reported for statistical significance (p < 0.01).
- The focal loss formulation with gradient detachment is mathematically well-justified.
- Two-stage transfer learning protocol is clearly described and reproducible.

### Weaknesses
- FER2013 does not provide subject identifiers, so image-level splitting is acknowledged — but the paper should explicitly state that this risks data leakage if augmented versions of the same image appear in both train and test folds.
- The CK+ test set has only 114 samples with 95% CI of ±8%, yet the paper presents per-class F1 values on CK+ (mentioned in text line 530). These per-class values on 114 samples are not presented in a table, making them unverifiable.
- No effect size reporting alongside p-values. Cohen's d or confidence intervals for the main comparisons would strengthen the statistical claims.

### Recommendations
- Add a sentence clarifying whether augmented images could leak across folds in the FER2013 CV setup.
- Report effect sizes (Cohen's d) for the main MobileNetV3+CSAM vs. baseline comparisons.

---

## 3. Evidence Sufficiency (Score: 84/100)

### Strengths
- Two datasets evaluated (FER2013: 35,887 images; CK+: 804 images) with different characteristics (grayscale vs. color, resolution differences).
- Comprehensive baseline comparison including VGG-16, ResNet-50, CNN+LSTM, 3D-CNN — both with standard and full-pipeline training.
- Efficiency analysis (Table 3) with inference time and energy consumption.
- Per-class analysis (Table 6) and confusion matrix (Figure 6).
- Grad-CAM qualitative analysis (Figure 7) with psychological interpretation.
- State-of-the-art comparison (Table 7).
- The limitations section is unusually thorough (8 limitations), which strengthens credibility.

### Weaknesses
- **CRITICAL: Data inconsistency — FPS numbers contradict each other.** Line 570 states "45 fps on RTX 3090" while line 582 states "833 fps on an RTX 3090 GPU." The efficiency table (Table 3) reports 1.2ms inference time, which corresponds to 1000/1.2 = 833 fps. The "45 fps" figure is factually incorrect and must be fixed.
- The CK+ cross-dataset evaluation is acknowledged as preliminary (114 test samples), but the claim "71.9% accuracy is comparable to within-dataset 70.5%" could mislead readers. The ±4.4% std and small sample size mean the CK+ result is far less reliable.
- No real classroom deployment or edge device benchmarking, despite the paper's emphasis on deployment. The Jetson Nano energy calculation (line 586) is theoretical.
- The abstract mentions "DAiSEE" in the task description but no DAiSEE results appear in the paper. (Note: DAiSEE experiments were run and showed 62.2% test accuracy / 43.6% F1, suggesting the emotion-to-engagement mapping does not transfer well to engagement-labeled data. The authors appear to have excluded these results.)

### Recommendations
- **Fix the FPS inconsistency**: Line 570 should state "833 fps" (or remove the specific fps claim and let Table 3 speak).
- Strengthen the CK+ cross-dataset interpretation by emphasizing the wide confidence interval.
- Consider adding a brief DAiSEE discussion in the limitations section to transparently acknowledge that the mapping was tested on engagement-specific data with poor results.

---

## 4. Argument Coherence (Score: 80/100)

### Strengths
- The paper follows a clear logical flow: motivation → theory → method → experiments → discussion → limitations.
- The theoretical grounding (Section 2.2) provides principled justification for design choices.
- The limitations section is honest and comprehensive.
- The training strategy confound is proactively addressed with full-pipeline baselines.

### Weaknesses
- **The FPS inconsistency (45 vs 833) directly contradicts the efficiency table and undermines the deployment argument.** This is the most impactful coherence issue.
- **Garbage LaTeX after \end{document}** (lines 687-688 contain duplicate table rows). While this doesn't affect the compiled PDF, it indicates sloppy file management.
- The claim "CSAM contributes 1.3--5.0% accuracy gain" (abstract) spans a very wide range. On MobileNetV3 it's +5.0%, on EfficientNet-B0 it's +1.3%. These are very different contributions and should be reported separately rather than as a range.
- Line 570 claims "45 fps" while the paper's own Table 3 shows 1.2ms = 833 fps. This contradiction appears in the Discussion section, which is the paper's interpretive backbone.

### Recommendations
- Fix the fps figure on line 570.
- Remove the garbage text after \end{document}.
- Report CSAM gains per backbone separately rather than as a misleading range.

---

## 5. Writing Quality (Score: 86/100)

### Strengths
- Professional academic tone throughout.
- Well-structured with clear section headings and logical progression.
- Tables and figures are well-designed and informative.
- The limitations section is a model of scholarly transparency.
- APA formatting is generally correct for MDPI Education Sciences.

### Weaknesses
- **Citation style inconsistency**: Lines 49, 90, 94, and 190 use `\cite{...}` where `\citep{...}` should be used for parenthetical citations in APA style. In APA, `\cite` produces textual citations (Author, Year) while `\citep` produces parenthetical ((Author, Year)). When the citation appears after a tilde at the end of a clause, `\citep` is appropriate.
- Some reference entries have incomplete fields (e.g., `dewan2019engagement` lacks volume/number; `biasutti2022surveillance` is never cited in the paper).
- The abbreviations table (line 670-681) includes NAS and SE but these are not prominent enough to warrant abbreviation listing.

### Recommendations
- Fix `\cite` → `\citep` on lines 49, 90, 94, 190.
- Remove uncited references or cite them.
- Clean up abbreviations table.

---

## Summary of Required Fixes (Priority Order)

1. **[CRITICAL]** Fix FPS inconsistency: line 570 "45 fps" → "833 fps" (matching 1.2ms from Table 3)
2. **[HIGH]** Remove garbage LaTeX after \end{document} (lines 687-688)
3. **[MEDIUM]** Fix \cite → \citep on lines 49, 90, 94, 190
4. **[MEDIUM]** Report CSAM gains per backbone separately in abstract
5. **[LOW]** Add effect size reporting (Cohen's d) for main comparisons
6. **[LOW]** Remove uncited reference `biasutti2022surveillance` or cite it

---

## Path to ≥90

To reach ≥90 overall, the paper needs:
- All data inconsistencies fixed (currently the biggest drag on Argument Coherence)
- The garbage text removed
- Citation style consistency
- Minor additions: effect sizes, augmented image leakage clarification

If these are addressed, the estimated post-fix scores would be:
- Originality: 78 (unchanged — structural, not fixable by edits)
- Methodological Rigor: 90 (with effect size addition)
- Evidence Sufficiency: 86 (with FPS fix and stronger CK+ interpretation)
- Argument Coherence: 90 (with all inconsistencies fixed)
- Writing Quality: 90 (with citation fixes)

Estimated post-fix overall: 78×0.20 + 90×0.25 + 86×0.25 + 90×0.15 + 90×0.15 = **86.9**

The Originality score is the structural ceiling (~78) and will not change without fundamental content additions. The paper is well-suited for Education Sciences but the contribution is incremental from a CV perspective.
