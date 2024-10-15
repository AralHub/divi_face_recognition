build:
	docker-compose build

start:
	docker-compose up -d

stop:
	docker-compose down

restart:
	docker-compose down
	docker-compose up -d

remove:
	docker-compose down -v

log:
	docker-compose logs -f
