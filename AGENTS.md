# Fashion-MNIST Classification Project - Agent Notes

## Project Overview
- **Dataset:** Fashion-MNIST (Zalando, 2017)
- **Task:** Compare CNN (MCNN15) vs Random Forest for fashion product classification
- **Performance:** CNN 93.63% test accuracy vs RF 87.52%
- **Report:** LaTeX academic report using XeLaTeX (Arial font)

## Key Technical Details

### Architecture
- **CNN:** MCNN15 (15-layer CNN) from Bhatnagar et al. (2017)
  - 3 groups of conv blocks with batch normalization
  - Training time: ~900s on GPU
  - Model size: 10MB
- **Random Forest:** scikit-learn
  - n_estimators=100, max_depth=100, criterion='entropy'
  - Training time: ~9.45s on CPU
  - Model size: 123MB

### Dataset
- 70,000 grayscale images (28×28 pixels)
- 10 balanced classes (6,000 train / 1,000 test per class)
- Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

### Project Structure
```
├── src/
│   ├── cnn/              # CNN training and evaluation
│   ├── random_forest/    # Random Forest training and evaluation
│   ├── common/           # Shared utilities (eval.py, paths.py, shared.py)
│   └── visualization/    # Dataset visualization scripts
├── plots/
│   ├── cnn/             # CNN confusion matrices, training curves, misclassified
│   ├── random_forest/   # RF confusion matrices, misclassified
│   └── dataset/         # Dataset sample visualizations
├── report/
│   ├── tex/             # LaTeX source files
│   │   ├── report.tex   # Main report file
│   │   └── references.bib  # Bibliography database (biblatex)
│   ├── task.md          # Assignment requirements
│   └── todo.md          # Agent task tracking
└── AGENTS.md            # This file
```

## Report Specifications

### Page Count Notes
- **Total PDF pages:** 11 pages (includes title page, TOC, references, appendix)
- **Body content:** Approximately 6.5 pages of actual report text
- **Target:** 7-10 pages for body content (currently slightly under)

### Compilation
```bash
cd report/tex
xelatex report.tex    # First pass
biber report          # Process bibliography (required for citations)
xelatex report.tex    # Second pass to resolve citations
xelatex report.tex    # Final pass for stability
```

**Note:** The report now uses `biblatex` with `biber` backend for APA-style citations. The `biber` step is required to process the bibliography database.

### Key Packages
- Document class: `article` with a4paper, 11pt
- Font: Arial (requires XeLaTeX)
- Bibliography: `biblatex` with APA style (requires `biber`)
- Required packages: `fontspec`, `graphicx`, `float`, `subcaption`, `booktabs`, `multirow`, `biblatex`

## Plot File Locations
All plots are referenced from `report/tex/report.tex` using relative paths:
- `../../plots/cnn/confusion_train.png`
- `../../plots/cnn/confusion_test.png`
- `../../plots/cnn/training.png`
- `../../plots/random_forest/confusion_train.png`
- `../../plots/random_forest/confusion_test.png`
- `../../plots/dataset/samples.png` (new)

## Running Scripts
Use `uv run` to execute Python modules:
```bash
uv run -m src.visualization.dataset_samples    # Generate dataset samples
uv run -m src.cnn.eval                         # Evaluate CNN
uv run -m src.random_forest.eval              # Evaluate Random Forest
```

## Class Names (Index Order)
```python
CLASS_NAMES = [
    "T-shirt/top",  # 0
    "Trouser",       # 1
    "Pullover",      # 2
    "Dress",         # 3
    "Coat",          # 4
    "Sandal",        # 5
    "Shirt",         # 6
    "Sneaker",       # 7
    "Bag",           # 8
    "Ankle boot",    # 9
]
```

## Bibliography Management

### Citation System
- **Backend:** `biber` (modern replacement for BibTeX)
- **Style:** APA 7th edition (`style=apa`)
- **Database:** `references.bib` in `report/tex/`

### Citation Commands
Use these LaTeX commands instead of hardcoding citations:
- **Parenthetical:** `\parencite{xiao2017}` → (Xiao et al., 2017)
- **Narrative:** `\textcite{xiao2017}` → Xiao et al. (2017)
- **Multiple:** `\parencite{xiao2017,bhatnagar2017}` → (Xiao et al., 2017; Bhatnagar et al., 2017)

### Adding New References
1. Add BibTeX entry to `report/tex/references.bib`
2. Use entry types: `@article`, `@inproceedings`, `@online`, `@book`
3. Run `biber report` after adding new citations
4. Recompile with `xelatex` (3 passes for stability)

## Common Issues
1. **Page count:** Currently ~6.5 pages body content (target: 7-10)
2. **Missing elements:**
   - Graphical abstract (Page 1)
   - Literature review (Page 2)
3. **Sample image orientation:** Dataset samples figure may need 90-degree rotation for better layout
4. **Citation errors:** If citations show as "[xiao2017]", run `biber report` and recompile

## Task Requirements Reference
See `report/task.md` for full assignment details including:
- Required report structure (Page 1-7 + Appendix)
- Mandatory sections (Abstract, Introduction, Methodology, Results, Analysis, Conclusion)
- Evaluation metrics (confusion matrices, precision/recall, training times)
- Page count: 7-10 pages + code
