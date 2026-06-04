# Peer Review Report — Round 1

**Paper Title:** Deep Learning for Student Engagement Detection: A Lightweight CNN Approach for Educational Environments  
**Review Date:** 2026-06-04  
**Reviewer:** Simulated Double-Blind Review

---

## Dimension Scores

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Originality | 20% | 72 | 14.4 |
| Methodological Rigor | 25% | 78 | 19.5 |
| Evidence Sufficiency | 25% | 80 | 20.0 |
| Argument Coherence | 15% | 82 | 12.3 |
| Writing Quality | 15% | 75 | 11.25 |
| **Overall** | **100%** | — | **77.45** |

---

## Detailed Review

### 1. Originality (72/100)

**Strengths:**
- The combination of MobileNetV3 and EfficientNet-B0 for engagement detection is a reasonable and practical approach
- The focus on edge deployment and practical applicability is valuable

**Weaknesses:**
- The novelty is incremental — applying existing lightweight architectures to engagement detection is not highly original
- Similar work has been done by Santoni et al. (2023) and others on DAiSEE
- The paper does not propose any novel architectural modification or training strategy

**Recommendations:**
- Consider adding a novel component (e.g., attention mechanism, loss function modification) to increase originality
- Clearly articulate what distinguishes this work from Santoni et al. (2023)
- Add a "Contribution" subsection in Introduction that more precisely states the novelty

### 2. Methodological Rigor (78/100)

**Strengths:**
- The experimental setup is well-described
- Ablation studies provide good insight into component contributions
- Transfer learning protocol is clearly specified

**Weaknesses:**
- The data preprocessing pipeline is not fully described (face detection method not specified)
- No cross-validation or statistical significance tests reported
- Hyperparameter selection process not documented (why these specific values?)
- The claim of "3.2× faster inference" needs clarification on hardware and batch size

**Recommendations:**
- Specify the face detection method (MTCNN, RetinaFace, etc.)
- Add 5-fold cross-validation with mean±std reporting
- Report statistical significance (paired t-test or Wilcoxon) for key comparisons
- Document hyperparameter search process

### 3. Evidence Sufficiency (80/100)

**Strengths:**
- Comprehensive comparison with multiple baselines
- Good ablation study design
- Per-class analysis provides useful insights

**Weaknesses:**
- Only one dataset (DAiSEE) used for evaluation
- No visualization of learned features (e.g., Grad-CAM)
- No qualitative examples of correct/incorrect predictions
- Some baseline results appear to be cited rather than reproduced

**Recommendations:**
- Add evaluation on at least one additional dataset (e.g., EmoReact, EmoBank)
- Include Grad-CAM visualizations showing what the model attends to
- Add qualitative examples showing model predictions vs. ground truth
- Reproduce baseline results under same conditions for fair comparison

### 4. Argument Coherence (82/100)

**Strengths:**
- Clear problem statement and motivation
- Logical flow from introduction to conclusion
- Good discussion of limitations

**Weaknesses:**
- The connection between "lightweight" and "deployable" could be strengthened with actual deployment experiments
- The claim about privacy benefits of edge deployment is mentioned but not substantiated

**Recommendations:**
- Add a deployment experiment on actual edge device (e.g., Jetson Nano)
- Strengthen the privacy argument with concrete analysis

### 5. Writing Quality (75/100)

**Strengths:**
- Generally clear and well-organized
- Good use of tables and figures

**Weaknesses:**
- Some sentences are overly long and complex
- The abstract is too long (300+ words, MDPI recommends 150-250)
- Some AI-typical phrases detected ("remarkable success", "significant computational challenges")
- Chinese abstract could be more concise

**Recommendations:**
- Shorten the abstract to 200 words
- Break long sentences into shorter ones
- Replace AI-typical phrases with more discipline-specific language
- Tighten the Chinese abstract

---

## Summary of Required Revisions

### Must Fix (Blocking):
1. **Shorten abstract** to 200 words (MDPI requirement)
2. **Add face detection method** specification in Method section
3. **Replace AI-typical phrases** throughout the paper

### Should Fix (Strong Recommendation):
4. Add cross-validation or statistical significance tests
5. Add Grad-CAM visualization or qualitative examples
6. Clarify inference speed measurement conditions
7. More clearly distinguish from Santoni et al. (2023)

### Nice to Have (Optional):
8. Add deployment experiment on edge device
9. Add evaluation on second dataset
10. Document hyperparameter search process

---

## Decision

**Recommendation: Minor Revision**

The paper presents a reasonable application of lightweight CNNs to engagement detection. The experimental results are solid, but the paper needs improvement in originality articulation, methodological documentation, and writing quality. The required revisions are achievable within one revision round.
