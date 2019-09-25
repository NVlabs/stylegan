dockerfile := "Dockerfile.gpu"
tag_name := "juniorxsound/stylegan:latest"
runtime := "--gpus all"

build:
	docker build -f ./$(dockerfile) -t $(tag_name) ./

shell:
	docker run $(runtime) -w /data --rm -it -v `pwd`:/data -t $(tag_name) /bin/bash

example:
	docker run $(runtime) -w /data --rm -it -v `pwd`:/data -t $(tag_name) python3 pretrained_example.py

jupyter:
	docker run $(runtime) -p 8888:8888 -w /data --rm -it -v `pwd`:/data -t juniorxsound/exit-stereo-metadata:latest jupyter notebook --allow-root \