# DefectSynth: Few-Shot Defective Image Generation by Modeling Shape and Appearance

## Environment Setup

```bash
conda env create -f environment.yml
conda activate DefectSynth
```

## Dataset and Checkpoint Preparation

1. **[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)** dataset (or a custom dataset with the same directory structure):
  Place it under `./training/MVTec_AD`.
2. **JSON Pair List**
  Each line should contain a JSON object with the following fields:`source`, `target`, `prompt`
   Save the file as: `./training/prompt.json`
3. **[Stable Diffusion v1.5 Base Checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)**:
  Download `v1-5-pruned.ckpt` and place it under: `./models`

---

## Training

---

### Validate Training Dataset


```bash
python validate_dataset.py \
  --prompt_json_path ./training/prompt.json \
  --data_root ./training/MVTec_AD \
  --check_count 5
```

---

### Train

```bash
python train.py \
  --prompt_json_path ./training/prompt.json \
  --data_root ./training/MVTec_AD
```

---

## Generation

### Stage 1: Generate Defect Masks

```bash
python mask_morph_gen.py \
  --folder_path ./training/MVTec_AD \
  --output_path ./outputs/masks_gen/MVTec_AD \
  --target_total 1000
```

---

### Stage 2: Generate Defect Images

```bash
python defect_image_gen.py \
  --txt_path ./training/name_defect.txt \
  --resume_path ./lightning_logs/MVTec_AD/last.ckpt
```

---

## License

Portions of this project are derived from or built upon third-party work, including
[ControlNet](https://github.com/lllyasviel/ControlNet),
[Latent Diffusion Models](https://github.com/CompVis/latent-diffusion),  
and [Stable Diffusion](https://github.com/Stability-AI).
Those components remain subject to their respective licenses and terms of use.

Stable Diffusion weights are **not** included in this repository. Users must download
checkpoints separately and comply with the
[Stable Diffusion license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).

---

## Citation

If you find our work useful in your research, please consider citing:

```bash
@article{zhao2026defectsynth,
  title     = {DefectSynth: Few-Shot Defective Image Generation by Modeling Shape and Appearance},
  author    = {Dexu Zhao, Xukun Qin, Xinghui Dong},
  journal   = {IEEE Transactions on Automation Science and Engineering},
  doi       = {10.1109/TASE.2026.3697519},
  year      = {2026},
  note      = {Accepted, to appear},
  publisher = {IEEE}
}
```
