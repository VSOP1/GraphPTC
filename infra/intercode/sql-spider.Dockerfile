FROM public.ecr.aws/docker/library/mysql:latest

ENV MYSQL_ROOT_PASSWORD="password"

ADD data/sql/spider/ic_spider_dbs.sql /docker-entrypoint-initdb.d/
