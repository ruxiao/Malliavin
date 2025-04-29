MallGSSN: Malliavin‑Gated State‑Space Networks
=============================================
A reference implementation accompanying the paper:
> **Malliavin‑Gated State‑Space Networks: Learnable Noise‑Retention for Linear‑Time Long‑Context Modelling**

Features
--------
* Malliavin pathwise gradient for learnable σ‑noise gates
* ResNet & Mamba backbones with interchangeable blocks
* Linear‑time memory footprint, optional external retrieval
* Out‑of‑box CIFAR‑100 example; long‑context synthetic tasks

Quick start
-----------
```bash
# create env & install
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt

# train CIFAR‑100 large config
bash scripts/train.sh configs/cifar100_large.yaml

# run unit tests
pytest -q
```
Full documentation is in `docs/` (to be written).
