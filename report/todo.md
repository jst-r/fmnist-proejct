# Report Enhancement TODO

## Current Priority: Dataset Description (Completed)
- [x] Plan dataset description expansion
- [x] Create visualization script for dataset samples
- [x] Generate sample images plot
- [x] Expand Introduction with comprehensive dataset description
- [x] Add sample images figure to Introduction
- [x] Reduce redundant dataset description in Methodology
- [x] Compile and verify LaTeX

## Immediate Fixes Needed
- [x] Rotate dataset samples image 90 degrees for better layout in report
  - **Completed:** Changed from 10×5 to 5×10 layout (5 rows × 10 columns)
  - Labels now appear on top row with 45-degree rotation
  - Better fits the 0.48\textwidth figure environment

## Future Tasks

### Graphical Abstract
- **Location:** Page 1, alongside textual abstract
- **Content:** Present motivation, method, results, conclusion visually
- **Options:**
  - Workflow diagram (input → CNN/RF → predictions)
  - Results comparison chart
  - Mini infographic showing key findings

### Literature Review
- **Location:** Page 2, after Introduction
- **Content:** Previous work on Fashion-MNIST classification
  - Key benchmark results from Xiao et al. (2017)
  - CNN architectures tested on Fashion-MNIST
  - Traditional ML approaches (Random Forest, SVM, etc.)
  - Performance comparisons from literature

### Code Attachment
- **Current:** Code in Appendix as listings
- **Task requirement:** "Attach a pdf of the python notebook after it run with all results. Alternatively, share the link to your repository."
- **Action needed:** Ensure repository link is provided or notebook PDF is attached

## Page Count Notes
- **Current body content:** ~6.5 pages (excluding title, TOC, references, appendix)
- **Target:** 7-10 pages for body content
- **Current total PDF:** 11 pages (includes all front matter and back matter)
