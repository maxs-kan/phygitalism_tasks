FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./miniconda.sh -b -p /app/conda && \
    rm ./miniconda.sh && \
    /app/conda/bin/conda env update -n base --file /app/environment.yml 
ENV PATH=/app/conda/bin:$PATH
EXPOSE 22 8501 
CMD /app/run.sh
