api:
	uvicorn dubstudio.main:app --host 127.0.0.1 --port 8080 --reload

web:
	cd web && npm run dev

test:
	pytest -q
