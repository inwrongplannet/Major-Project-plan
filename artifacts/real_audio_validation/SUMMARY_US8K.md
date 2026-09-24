# UrbanSound8K Validation Summary

## Setup
To address the lack of gunshot clips in the initial ESC-50 validation and to expand the real-audio sample pool, the dataset **UrbanSound8K** was integrated into the validation pipeline.

**Dataset Citation:**
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

**Important Caveat on Validation Splits:**
UrbanSound8K is natively provided as 10 pre-defined folds (fold1 through fold10), and the authors strongly recommend performing 10-fold cross-validation when comparing models on this dataset. However, to integrate this dataset seamlessly with the existing ESC-50 + synthetic pipeline (ARCH_3), this project utilizes **random stratified splits** rather than the official folds. While this breaks strict comparability with published UrbanSound8K benchmarks, it aligns with our project's goal of evaluating generalizability rather than competing on UrbanSound8K specifically.

## Experiment 3: Zero-Shot (Gunshot Evaluation)
The zero-shot evaluation on the UrbanSound8K dataset yields the first real-audio result for the `gunshot` class in this project. 
When the purely synthetic-trained model was evaluated on the mapped UrbanSound8K categories without any retraining, the overall argmax accuracy was **7.9%**.
For the `gunshot` category specifically (n = 374 clips), the alert rate was **9.1%**. 

## Multi-Seed Combined Retraining (Experiment 7)
To ensure robustness, the retraining experiment combining Synthetic, ESC-50, and UrbanSound8K data was run across 5 random seeds (42, 43, 44, 45, 46). 

The mean accuracy on real-audio test clips across the 5 seeds is **65.7% ± 4.1%** standard deviation. The results are highly stable across iterations.

**Per-Seed Breakdown:**
- **Seed 42:** 64.7% (n=218)
- **Seed 43:** 68.3% (n=218)
- **Seed 44:** 71.1% (n=218)
- **Seed 45:** 60.7% (n=219)
- **Seed 46:** 63.6% (n=217)

## Cross-Dataset Comparison Table

| Experiment | Real-audio accuracy | n | Gunshot covered? |
|---|---|---|---|
| ESC-50 zero-shot (earlier plan) | 15.6% | 480 | No |
| ESC-50 retrained, real-only (earlier plan) | 67.1% | 146 | No |
| UrbanSound8K zero-shot, gunshot only (T4) | 9.1% | 374 | Yes |
| ESC-50 + UrbanSound8K retrained, real-only, multi-seed (T7) | 65.7% ± 4.1% | 217-219 | Yes |

## Limitations
- **Chainsaw Category:** The `chainsaw` category remains unrepresented in the real-audio validation. Neither ESC-50 nor UrbanSound8K contains chainsaw samples, meaning it remains a synthetic-only class in our evaluation.
- **Ambient Category Semantics:** The `ambient` class in our taxonomy targets forest/eco-ambience. However, due to the nature of UrbanSound8K, the tested clips are urban ambience (e.g., street music, air conditioner).
- **Non-Standard Folds:** As mentioned in the setup, the official UrbanSound8K 10-fold cross-validation protocol was not utilized, preventing direct comparison with literature.
