all: dataset prepare visualizations features train

dataset:
	gdown -O data/raw/flights_raw.csv.gz https://drive.google.com/u/0/uc?id=1gVjoaf_MK-PY3t-fMza1u1A5o4uGwAof

prepare: src/01_prepare_data.py
	python src/01_prepare_data.py

visualizations: src/02_make_visualizations.py
	python src/02_make_visualizations.py

features: src/03_build_features.py
	python src/03_build_features.py
	
train: src/04_train_model.py
	python src/04_train_model.py --model rf --on valid > reports/results_rf_valid.txt
	python src/04_train_model.py --model rf --on test -o rf.pkl > reports/results_rf_test.txt
	cat reports/results_rf_test.txt

modelB: src/03_build_features.py src/04_train_model.py
	python src/03_build_features.py --min_drop 0
	python src/04_train_model.py --model rf --on valid > reports/results_rf_B_valid.txt
	python src/04_train_model.py --model rf --on test -o rf_B.pkl > reports/results_rf_B_test.txt
	cat reports/results_rf_B_test.txt