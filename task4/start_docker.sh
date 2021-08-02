#/bin/bash
docker build -t maxskan/vis .
docker run -d --rm -it \
	   -p 8501:8501 \
	   -v /home/kan/d/dataset-v2:/data \
           --name kan_vis maxskan/vis 
