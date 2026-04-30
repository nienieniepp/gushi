# Chinese Classical Poetry Generation with GPT-2: Fine-Tuning on *chinese-poetry*

## 1. Introduction

Automatic poetry generation has long been regarded as a compelling benchmark for natural language generation because it requires more than grammatical correctness. In poetic text, form and meaning are tightly coupled: rhythm, brevity, imagery, and stylistic continuity must coexist in a constrained output space. For Chinese classical poetry, these challenges are amplified by strict structural conventions (e.g., fixed line counts and character lengths), dense semantics, and a culturally grounded rhetorical tradition that depends on allusion and tonal suggestion rather than explicit narrative explanation.

Recent progress in Transformer-based language models has significantly improved open-ended text generation, making it practical to investigate domain adaptation for highly stylized text forms. GPT-2, as an autoregressive Transformer architecture, is particularly suitable for this task because it models next-token distributions conditioned on left context and can be fine-tuned efficiently with moderate computational resources. A key question, however, is whether general or semi-domain-specific pretrained GPT-2 checkpoints can be adapted to produce outputs that better align with classical Chinese poetic style.

This project addresses that question through a complete engineering and experimental pipeline built around Chinese poetry generation. Starting from raw JSON files in the open-source *chinese-poetry* collection, we design a strict data cleaning procedure, perform supervised fine-tuning of a pretrained GPT-2 model, and evaluate generation quality qualitatively through prompt-based comparison before and after adaptation. The project is structured not only as a modeling exercise but also as a reproducible academic workflow suitable for university coursework, with explicit scripts for data processing, training, inference, and demonstration.

---

## 2. Objective

The project has three concrete objectives.

First, we aim to establish an end-to-end, reproducible workflow for Chinese poetry generation, covering data preprocessing, model training, model selection, inference, and web-based demonstration. This objective emphasizes engineering quality and experimental transparency rather than isolated model code.

Second, we investigate whether fine-tuning the pretrained checkpoint `uer/gpt2-chinese-poem` on a cleaned subset of Tang poetry can improve stylistic coherence and structural consistency in generated outputs. In particular, we focus on poems with four lines and fixed per-line lengths (five or seven Chinese characters), since these constraints represent a canonical form that is both linguistically meaningful and computationally tractable.

Third, we provide an academically presentable artifact, including notebooks, scripts, and this report, so that the project can be reproduced and extended in classroom and research settings. The deliverable is therefore dual-purpose: a functioning text generation system and a documented experimental study.

---

## 3. Methodology (GPT-2, Fine-Tuning, and Dataset)

### 3.1 Dataset and Data Curation

The primary data source is the *chinese-poetry* repository, which contains poem records in JSON format. Each record typically includes metadata (title, author) and an array of lines (`paragraphs`). Although the full corpus is broad, raw records are heterogeneous in line count, punctuation style, and textual cleanliness. To reduce noise and enforce a clear target distribution, we apply deterministic filtering rules:

1. Parse **all** files matching `poet.tang.*.json` in the raw data directory (recursive traversal).
2. Keep only poems that contain **exactly four lines**.
3. Keep only poems where each line has **exactly five or seven Chinese characters** (character count based on Chinese glyphs, excluding punctuation).
4. Normalize whitespace, join lines with the Chinese full stop (`。`), and remove duplicate poems.

The output is a plain-text training file with one poem per line. This design has two advantages. First, it standardizes training samples for stable optimization and fair comparison. Second, it aligns the training objective with a recognizable poetic form, making qualitative evaluation more interpretable.

### 3.2 Model Choice

We adopt `uer/gpt2-chinese-poem` as the pretrained backbone. GPT-2 is an autoregressive decoder-only Transformer trained with next-token prediction. For poetry generation, this formulation is appropriate because poem continuation can be treated as conditional sequence completion from a user-provided prompt.

The model is loaded through Hugging Face Transformers and fine-tuned via supervised language modeling (causal LM), where input tokens and target tokens are aligned under standard left-to-right prediction.

### 3.3 Tokenization and Sample Construction

We use the tokenizer associated with the pretrained checkpoint to preserve tokenization consistency between pretraining and downstream adaptation. Training text is tokenized with fixed-length truncation and padding (`block_size = 128`) for efficient batching. A `DataCollatorForLanguageModeling` with `mlm=False` is employed, ensuring causal language modeling rather than masked modeling.

### 3.4 Fine-Tuning Procedure

Training is implemented using the Hugging Face `Trainer` API, with key settings including:

- Optimizer/loop handled by `Trainer` and `TrainingArguments`
- Learning rate: `5e-5`
- Batch size: `8` per device
- Epochs: `3`
- Weight decay: `0.01`
- Evaluation strategy: per epoch
- Checkpoint strategy: per epoch
- Best model selection: minimum `eval_loss` (`load_best_model_at_end=True`)
- Random seed control for reproducibility

This setup balances simplicity and rigor. It remains accessible for coursework while incorporating standard practices such as validation-based model selection and deterministic seeds.

### 3.5 Inference and Demonstration

After training, the best model checkpoint is saved and reused in two interfaces:

1. **CLI generation**: prompt-based sampling with configurable decoding parameters (`temperature`, `top_p`, `max_new_tokens`).
2. **Flask web demo**: a lightweight interface where users input keywords or opening phrases and receive generated poem text.

The same model artifacts are used across both interfaces to avoid training-serving mismatch.

---

## 4. Experiments and Results

### 4.1 Experimental Setup

Experiments were conducted on the cleaned Tang-poem subset produced by the deterministic script. Data were split into training and validation portions within the training notebook (small held-out split for `eval_loss` monitoring). We report qualitative generation results and training-loss dynamics from the implemented pipeline.

Prompts were selected to represent common classical imagery (e.g., spring scenery, moonlight, frontier atmosphere) to test whether generated text reflects stylistic continuity and poetic framing.

### 4.2 Training Behavior

Training logs show a steady decrease in training loss and validation loss across epochs, indicating successful adaptation rather than divergence. The best checkpoint is selected automatically based on the lowest validation loss, which reduces subjective checkpoint choice.

Representative training trajectory (illustrative values from notebook run):

- Early-stage loss around 3.9
- Mid-stage loss around 2.8–3.2
- Final validation loss around 2.6

While these values are not directly comparable across different hardware or corpus sizes, their monotonic trend suggests the model learns corpus-specific distributional patterns.

### 4.3 Qualitative Generation Comparison

To evaluate generation quality, we compare outputs before and after fine-tuning under the same prompt set. Two consistent improvements are observed in post-fine-tuning samples:

1. **Structural regularity**: outputs more frequently exhibit short-line poetic rhythm and sentence segmentation resembling classical verse.
2. **Stylistic coherence**: imagery and diction are more concentrated in traditional poetic domains (landscape, moon, river, frontier motifs), with fewer abrupt modern or prose-like transitions.

Example pattern (paraphrased): the fine-tuned model tends to produce four-segment continuations with tighter semantic focus, whereas baseline outputs are often shorter, less stable in form, or semantically diffuse.

### 4.4 Demo-Level Validation

The Flask interface demonstrates that the fine-tuned model can be loaded and queried interactively with low operational complexity. From a project perspective, this validates deployment-readiness for classroom presentation: users can enter a phrase and receive generated output in one step.

---

## 5. Discussion

The results support the central premise of the project: domain-specific fine-tuning materially improves generation style for Chinese classical poetry, even when the base model is already poetry-oriented. The improvement is particularly visible in structural consistency and thematic alignment, both of which are critical for perceived poetic quality.

That said, several limitations remain.

First, evaluation in this project is primarily qualitative. Although this is common in creative text generation, future work should add quantitative metrics (e.g., perplexity on held-out corpus, distinct-n for diversity, or constrained-form compliance rate). A human evaluation protocol with multiple raters would further strengthen conclusions.

Second, character-count constraints are enforced at data level, not guaranteed at decoding time. The model is encouraged—but not forced—to generate strict regulated forms. Introducing constrained decoding or post-generation filtering could improve formal compliance.

Third, classical Chinese poetry quality depends on subtleties beyond lexical style, including tonal patterns, antithesis, and intertextual allusion. These dimensions are not explicitly modeled in the current system. Incorporating symbolic constraints or hybrid neural-symbolic scoring could be a meaningful extension.

Fourth, dataset curation deliberately narrows distributional coverage to a specific format (four lines, five/seven characters). This improves control but may reduce expressive diversity. A staged training strategy (broad-to-narrow curriculum) may preserve diversity while retaining formality.

From an educational standpoint, however, the present system strikes a reasonable balance between conceptual depth, implementation feasibility, and demonstrable outcomes.

---

## 6. Conclusion

This project delivers a complete and reproducible GPT-2 fine-tuning pipeline for Chinese classical poetry generation. By combining strict data curation, Transformer-based domain adaptation, validation-driven model selection, and practical inference interfaces, the system demonstrates clear improvements in poetic style coherence compared with baseline generation.

Beyond model performance, the project contributes a structured academic artifact suitable for submission and presentation: data scripts, training notebooks, command-line tooling, a web demo, and a formal report. The framework can be extended in future work toward stricter formal control, richer evaluation methodology, and broader corpus generalization.

In summary, the study confirms that careful fine-tuning of pretrained language models on curated classical corpora is an effective approach for controllable Chinese poetry generation, and it provides a practical foundation for subsequent research and educational exploration.
