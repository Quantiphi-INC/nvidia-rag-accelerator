<div class="row">
  <div class="column">
    <img src="./media/NVIDIA-logo.png" alt="NVIDIA-LOGO" style="width:10%">
  </div>
  <div class="column">
    <img src="./media/nvidia-elite-partner-badge.png" alt="ELITE-PARTNER-LOGO" style="width:10%">
  </div>
  <div class="column">
    <img src="./media/Q-logo.png" alt="Q-LOGO" style="width:10%">
  </div>
</div>

## Architecture
<img src='./media/architecture.png' alt='architecture' style='width:80%'/>

## Tech
# Steps to setup for local development
- Clone the repo
- LLM
    - update tensorrtllm_backend submodules
    - download and create engine file for LLM based on the architecture
    - configure triton inference server
    - start triton inference server
- start docker compose services
    - backend
    - frontend
    - milvus-db
- check docker network
- hit ingest endpoint
- navigate to UI
- test application

# Setting up Project locally

Clone the repo with `git clone` which would contain **tensorrtllm_backend** as a submodule. Step into the folder to update the submodule dependencies.

To start the service run `docker compose up -d`. This would start backend, frontend and Milvus vector DB as microservice.
Backend would also mount './dataset' folder inside the container. To index the data into the vector database hit the /ingest_data endpoint with the dataset folder. This would chunk the data and index the same for RAG Pipeline.

## To Start LLM Service
To update the submodule dependencies run the following commands

```
cd tensorrtllm_backend/
git lfs install
git lfs pull
git submodule update --init --recursive
```
In the below examples we show the commands for running a gated model. We would need the huggingface token to be passed in the env variable of the below command. We can also adjust the GPU that is going to be used for triton inference server. In our case we are using gpu rank 1 for running it.

Running this docker container will add the container to the same network as other microservices, so that they can communicate with each other.

```
docker run --rm -it --env HF_TOKEN=hf_* --name=triton_server --network rag_accelerator -p8000:8000 -p8001:8001 -p8002:8002 --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus '"device=1"' -v ./tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash
```

### Install Llama requirements
The following commands are to be executed inside the docker container that we started in the above. Command assumes that we already have the engine file for the model that we are deploying. Follow this [document to create the engine file](https://github.com/NVIDIA/TensorRT-LLM#quick-start). We have to adjust the commands based on the model that we are using and the location where we put the engine files. This example assumes that we have 2 folders
- tensorrtllm_backend/tensorrt_llm/examples/llama/Llama-2-13b-chat-hf
- tensorrtllm_backend/tensorrt_llm/examples/llama/Llama-2-13b-chat-hf-engine

First one for the hugging face model and the second one for storing the engine file created by TensorRT-LLM.

Login with HF_TOKEN inside the container using `huggingface-cli login --token $HF_TOKEN` 

Install the requirements of llama model.
`pip install -r /tensorrtllm_backend/tensorrt_llm/examples/llama/requirements.txt && pip install protobuf`

### Copy config
`cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.`

### configuration of triton inference server
```
sed -i 's#${tokenizer_dir}#meta-llama/Llama-2-13b-hf#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt && \
sed -i 's#${tokenizer_type}#llama#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt && \
sed -i 's#${tokenizer_dir}#meta-llama/Llama-2-13b-hf#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt && \
sed -i 's#${tokenizer_type}#llama#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt && \
sed -i 's#${decoupled_mode}#true#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt && \
sed -i 's#${engine_dir}#/tensorrtllm_backend/tensorrt_llm/examples/llama/Llama-2-13b-chat-hf-engine/1-gpu/#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
```

### Start triton server
Run the command to start the triton server: 
`tritonserver --model-repository=/opt/tritonserver/inflight_batcher_llm`

 