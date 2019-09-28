dockerfile := "Dockerfile.gpu"
tag_name := "juniorxsound/stylegan:latest"

build:
	docker build -f ./$(dockerfile) -t $(tag_name) ./

shell:
	docker run --gpus all -w /data --rm -it -v `pwd`:/data -t $(tag_name) /bin/bash

example:
	docker run --gpus all -w /data --rm -it -v `pwd`:/data -t $(tag_name) python3 pretrained_example.py

jupyter:
	docker run --gpus all -p 8888:8888 -w /data --rm -it -v `pwd`:/data -t $(tag_name) jupyter notebook ./notebooks --ip=0.0.0.0 --allow-root \