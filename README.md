# Iris Recognition - Biometric Identification System

Complete implementation of iris recognition from segmentation through encoding to evaluation on MMU Iris Dataset.

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/yourusername/iris-recognition.git
cd iris-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install project
pip install -e .

# Run iris recognition pipeline on MMU dataset
python src/iris/main.py --config config/config.json run

# Run unit tests
python -m unittest discover -s tests -p "test_*.py"
```

## Project Structure

```
.
├── src/iris/
│   ├── pipeline.py          # Main pipeline: preprocessing → segmentation → encoding → matching → evaluation
│   ├── config.py            # Configuration loading
│   ├── errors.py            # Custom exceptions
│   ├── main.py              # CLI entry point
│   └── ...
├── tests/
│   ├── test_pipeline_utils.py  # Unit tests (9 tests, all passing)
│   └── ...
├── config/
│   ├── config.json          # MMU configuration (main)
│   └── default_config.json  # Default/fallback config
├── sprawozdanie/
│   └── main.pdf             # Final report (compiled LaTeX)
├── pyproject.toml           # Project metadata
└── README.md
```

## What This Project Does

**Iris Recognition Pipeline** (Etapy 2-8):
1. **Preprocessing**: Grayscale conversion, binarization
2. **Pupil Segmentation**: Circle detection with morphological operations + HoughCircles fallback
3. **Iris Segmentation**: Radial gradient analysis to find outer iris boundary
4. **Normalization**: Polar-to-rectangular mapping (rubber sheet model)
5. **Radial Bands**: Division into 8 radial sectors
6. **Gabor Encoding**: Complex filter response → 2-bit iris code
7. **Matching**: Masked Hamming distance with angular shift compensation
8. **Evaluation**: FAR/FRR/EER + Leave-One-Out identification accuracy

## Dataset

**MMU Iris Dataset** (450 images):
- 45 subjects
- 2 eyes per subject (treated as separate classes)
- 5 images per eye
- **Total: 90 iris classes**

## Final Results

Evaluated on full MMU Iris Dataset:

| Metric | Value |
|--------|-------|
| Genuine pairs | 900 |
| Impostor pairs | 100,125 |
| Mean genuine distance | 0.1781 |
| Mean impostor distance | 0.2272 |
| **EER** | **0.2066** |
| **Identification Accuracy (LOO)** | **85.78%** |

## Report

Full project report available in: `sprawozdanie/main.pdf`

Contains:
- Detailed algorithm descriptions
- Implementation notes and fixes
- Qualitative results (segmentation, normalization examples)
- Quantitative metrics on MMU dataset
- Conclusions

## Requirements & Approach

✓ Complete iris recognition system (preprocessing → evaluation)  
✓ Daugman-style iris coding with Gabor wavelets  
✓ Full validation on MMU Iris Dataset  
✓ FAR/FRR/EER computation  
✓ Leave-One-Out identification accuracy  
✓ 9 unit tests (all passing)  

## Key Implementation Details

- **Gabor Frequency**: π/128 ≈ 0.0245 (per specification)
- **8 Radial Bands** with 128 angular samples per band
- **Angular exclusion**: Top (20°) and bottom (20°) to handle eyelids/lashes
- **Angular shift compensation**: ±8 pixel shifts during matching
- **Robust pupil segmentation**: Morphological + percentile-based fallback
- **Proper MMU handling**: Left/right eyes as separate classes

## Testing

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Output: 9 tests OK ✓
```

## License

Educational project - Biometric Systems course

## Authors

Aleksander Brandt, Kamila Wieckowksa
