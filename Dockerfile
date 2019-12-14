FROM python:3.7-alpine as base
FROM base as builder

LABEL maintainer="Jacob Miller (jake_son@yahoo.com)"

COPY channelizer/requirements.txt /

RUN pip install --install-option="--prefix=/install" -r /requirements.txt
FROM base
COPY --from=builder /install /usr/local

COPY channelizer/ /channelizer

WORKDIR /channelizer

