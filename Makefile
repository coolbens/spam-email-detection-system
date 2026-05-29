install:
	pip install -r requirements.txt

run:
	streamlit run app/app.py

train:
	python -m model.train

fetch-data:
	python scripts/fetch_real_dataset.py

docker-build:
	docker build -t spam-detector .

docker-run:
	docker run -p 8501:8501 spam-detector
