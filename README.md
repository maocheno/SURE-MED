# Sure‑Med

<img width="621" height="518" alt="image" src="https://github.com/user-attachments/assets/12cdeaa0-05f2-41b5-bb6c-8eea1d801328" />

[Full Paper Path](https://arxiv.org/abs/2508.01693)
Automated medical report generation (MRG) holds great promise for reducing the heavy workload of radiologists. However, its clinical deployment is hindered by three major sources of uncertainty. First, visual uncertainty, caused by noisy or incorrect view annotations, compromises feature extraction. Second, label distribution uncertainty, stemming from long-tailed disease prevalence, biases models against rare but clinically critical conditions. Third, contextual uncertainty, introduced by unverified historical reports, often leads to factual hallucinations. These challenges collectively limit the reliability and clinical trustworthiness of MRG systems. To address these issues, we propose SURE-Med, a unified framework that systematically reduces uncertainty across three critical dimensions: visual, distributional, and contextual. To mitigate visual uncertainty, a Frontal-Aware View Repair Resampling module corrects view annotation errors and adaptively selects informative features from supplementary views. To tackle label distribution uncertainty, we introduce a Token Sensitive Learning objective that enhances the modeling of critical diagnostic sentences while reweighting underrepresented diagnostic terms, thereby improving sensitivity to infrequent conditions. To reduce contextual uncertainty, our Contextual Evidence Filter validates and selectively incorporates prior information that aligns with the current image, effectively suppressing hallucinations. Extensive experiments on the MIMIC-CXR and IU-Xray benchmarks demonstrate that SURE-Med achieves state-of-the-art performance. By holistically reducing uncertainty across multiple input modalities, SURE-Med sets a new benchmark for reliability in medical report generation and offers a robust step toward trustworthy clinical decision support.

## 📦 Pretrained Models

| Model Name                    | Description                                                 | Download Link                                                                 |
|-------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------------------|
| **XraySiglip**                  | ViT‑based feature extractor for chest X‑rays                | [HuggingFace](https://huggingface.co/StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli)                         |
| **CLIP‑Prior‑Filter**         | CLIP model fine‑tuned for historical report filtering       | [clip](https://stanfordmedicine.app.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml)                |
| **Vicuna**  | text decoder    | [HuggingFace](https://huggingface.co/lmsys/vicuna-7b-v1.5)                    |


## ⚙️ Installation & Usage

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/Focus-Med.git
   cd SURE-MED
## 📂 Checkpoints
[Checkpoint](https://huggingface.co/maoche/Sure-med/tree/main)  

##  More case study
<img width="1681" height="1168" alt="image" src="https://github.com/user-attachments/assets/37cb4462-9e94-456c-a5db-345554e82e9b" />

