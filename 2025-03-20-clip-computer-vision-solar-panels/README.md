# CLIP Computer Vision for Solar Panels

This project demonstrates using CLIP (Contrastive Language-Image Pre-training) for solar panel detection and analysis.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # CLIP analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Images directory
- CLIP model selection
- Detection threshold
- Output settings

## CLIP Model

CLIP (Contrastive Language-Image Pre-training):
- Zero-shot image classification
- Text-image matching
- Pre-trained on large dataset
- No fine-tuning required

## Caveats

- Requires image files in specified directory.
- CLIP model downloads on first use (requires internet).
- Full implementation requires transformers library setup.
