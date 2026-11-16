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

eeg-features-v2:
	nice -n 10 $(PY) modules/eeg_dementia/src/features_v2.py

eeg-nested:
	nice -n 10 $(PY) modules/eeg_dementia/src/evaluate_v2.py

eeg-cnn:
	nice -n 10 $(PY) modules/eeg_dementia/src/cnn.py

data-photic:
	bash scripts/download_ds006036.sh

eeg-features-v3:
	nice -n 10 $(PY) modules/eeg_dementia/src/features_v3.py
	nice -n 10 $(PY) modules/eeg_dementia/src/features_v2.py --root data/ds006036 --deriv derivatives/eeglab --task photomark --out data/features_v2_photic.parquet

eeg-nested-v3:
	nice -n 10 $(PY) modules/eeg_dementia/src/evaluate_v3.py --mode nested

eeg-nested-v3-photic:
	nice -n 10 $(PY) modules/eeg_dementia/src/evaluate_v3.py --mode nested --photic
