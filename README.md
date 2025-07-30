# Focus‑Med

A unified framework for reducing uncertainty in medical report generation.

## 📦 Pretrained Models

| Model Name                    | Description                                                 | Download Link                                                                 |
|-------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------------------|
| **XraySiglip**                  | ViT‑based feature extractor for chest X‑rays                | [HuggingFace](https://huggingface.co/StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli)                         |
| **CLIP‑Prior‑Filter**         | CLIP model fine‑tuned for historical report filtering       | [clip](https://huggingface.co/maoche/clip-prior-filter)                |
| **FOCUS‑Med Multimodal Gen**  | Pretrained multimodal report generator used in FOCUS‑Med    | [HuggingFace](https://huggingface.co/maoche/focus-med-gen)                    |

> **Tip:** You can add badges above the table for quick access, for example:
> ```markdown
> [![CheXbert](https://huggingface.co/maoche/chexbert/badge.svg)](https://huggingface.co/maoche/chexbert)
> ```

## ⚙️ Installation & Usage

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/Focus-Med.git
   cd Focus-Med
