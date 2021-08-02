#/bin/bash
docker build -t maxskan/vis .
docker run --runtime=nvidia -d --rm -it \
	   -p 8501:8501 \
           --name kan_vis maxskan/vis 
