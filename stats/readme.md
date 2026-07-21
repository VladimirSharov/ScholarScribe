# Dataset Statistics Manifest

This folder contains computed statistics for the current dataset version.

## Files

- `stats_summary.json`: Main summary of tag/text stats (produced by `scripts/compute_stats.py`)
- `tags/subject_tag_counts.json`: Frequency dictionary of all subject tags
- `tags/subject_rare_tags.json`: List of subject tags with only 1 occurrence
- `tags/additional_tag_counts.json`: Frequency dictionary of all additional tags
- `tags/additional_rare_tags.json`: List of additional tags with only 1 occurrence

## Notes

- Text lengths are computed as **number of whitespace-separated words**
- Language distribution is taken from `"abstract_language"` field
- Rare tags may be candidates for pruning or clustering

## Length Metrics

Both word and character (symbol) lengths are computed for:

- Thesis titles
- Abstracts

Values reported:
- Minimum
- Maximum
- Average
- Count of entities

You're moving forward — even if tired or lazy, you're on track. Let’s clean up **what’s been done**, **what matters**, and exactly **how to handle the input for XLM-R**.

---

## ✅ What’s Already Done

### ✅ Preprocessing:

* ✅ Loaded dataset from `data/full_dataset.json`
* ✅ Combined `subject_tags + additional_tags` → `combined_tags`
* ✅ Built tag vocabulary (tags with freq ≥ 5)
* ✅ Filtered samples without valid tags
* ✅ Split into `train/val/test` with 80/10/10
* ✅ Wrote files to `data_split_min5/`

  * `train.json`, `val.json`, `test.json`
  * `tag_vocab.json`

---

## 📌 What’s Next

1. Clean & tokenize the input (for XLM-R)
2. Train multi-label classifier
3. Evaluate results
4. Adjust (cutoff, sampling, augmentation...)

---

## ❓ Questions & Recommendations

### ❓ Should I remove `\r`, `\n`, etc. from abstract?

✅ Yes, normalize for clean input.

```python
def clean_text(text):
    return text.replace("\r", " ").replace("\n", " ").strip()
```

Also recommended:

* Replace multiple spaces with one
* Remove non-printable chars (optional)
* Don't lower/uppercase — XLM-R handles case

---

### ❓ What should I NOT do to text fields?

**Don’t:**

* Lowercase everything
* Lemmatize/stem
* Remove stopwords
* Translate or auto-correct
* Over-engineer it — XLM-R does subword tokenization

Just **clean noise**, not **semantics**.

---

### ❓ How should the input look for XLM-R?

You're doing **multi-label classification**, so:

#### ✅ Input

Use **abstract + title**, like:

```text
Opettajan ja oppilaan välisten ymmärryksen... [TITLE]
Vuorovaikutuksessa... [ABSTRACT]
```

Can join like:

```python
text_input = title + "\n\n" + abstract
```

#### ✅ Output

Binary vector per tag (sigmoid per label):

* For 3000 possible tags → output shape: `[batch_size, 3000]`
* Each entry: 1 if tag applies, 0 if not

---

### 🔧 XLM-R Setup Example (HuggingFace)

**Tokenizer:**

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

inputs = tokenizer(
    text_input,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

**Model:**

```python
from transformers import XLMRobertaModel
import torch.nn as nn

class XLMRMultiLabel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.base = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.base.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(self.dropout(pooled))
        return logits
```

**Loss Function:**

```python
torch.nn.BCEWithLogitsLoss()
```

---

## 🛠️ What You Should Do Now

1. Clean text (`\n`, `\r`, extra spaces)
2. Decide input: title + abstract, just abstract?
3. Prepare a dataset loader
4. Build the training loop or use `Trainer` from HuggingFace
5. Start training!

---

## 💤 You Can Stop Here for Now

You’ve done the split. Tomorrow:

* Start by prepping tokenizer + input pipeline
* Then model
* Then experiment

Want me to prepare:

* A **starter training script** (PyTorch + HuggingFace)?
* A **trainer** with early stopping, checkpointing?

Just confirm, and I’ll prepare it.

### `check_label_coverage.py`

Location: `scripts/check_label_coverage.py`

Purpose: 
Checks whether all tags in the validation and test splits are present in the training split. Ensures no zero-shot labels appear during evaluation, which would invalidate performance metrics.

Result for dataset `data_split_fi_eng_min5_abs40`:
- 0% of validation/test tags are unseen in training.
- Label coverage is complete. Evaluation setup is valid.