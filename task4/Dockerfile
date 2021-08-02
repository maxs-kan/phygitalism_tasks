FROM ubuntu:18.04 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	ca-certificates \
        curl
WORKDIR /app
COPY environment.yml .
RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./miniconda.sh -b -p /app/conda && \
    /app/conda/bin/conda env update -n base --file /app/environment.yml 



FROM ubuntu:18.04
EXPOSE 8501 
WORKDIR /app
COPY --from=build /app/conda/bin /app/conda/bin
COPY --from=build /app/conda/lib /app/conda/lib
ENV PATH=/app/conda/bin:$PATH
COPY src/* /app/
ENTRYPOINT ["streamlit", "run", "vis.py"]
