FROM ubuntu:18.04 as base
LABEL maintainer="Jacob Miller (jake_son@yahoo.com)"

RUN apt update && apt -y install \
    python3

# FROM base as builder

COPY channelizer/requirements.txt /

RUN apt -y install \
    python3-pip && \
    pip3 install -r /requirements.txt

# FROM base
# COPY --from=builder /install /usr/local

COPY channelizer/ /channelizer

WORKDIR /channelizer
