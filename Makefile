api:
	uvicorn dubstudio.main:app --host 0.0.0.0 --port 8080 --reload

web:
	cd web && npm run dev

web-build:
	cd web && npm install && npm run build

test:
	pytest -q

docker:
	docker compose up --build

docker-d:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
