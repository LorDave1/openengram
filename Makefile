PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1

setup:
	uv venv -p 3.12 .venv && uv pip install --python $(PY) -r requirements.txt

data:
	bash scripts/download_ds004504.sh

eeg-features:
	nice -n 10 $(PY) modules/eeg_dementia/src/features.py

eeg-baseline:
	nice -n 10 $(PY) modules/eeg_dementia/src/baseline.py
