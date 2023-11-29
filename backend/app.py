from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, validator
from typing import Optional, Union
from pathlib import Path
import os, re
from loader.data_loader import pipeline
from retriever.make_retrieve import get_context
from langchain.prompts import PromptTemplate
import requests
import json
import yaml
import argparse
from config import config_ns

app = FastAPI()
prompt = PromptTemplate.from_template(config_ns.prompt)

## SCHEMAS
class IngestData(BaseModel):
    path: str

    @validator("path")
    def path_must_be_directory(cls, v):
        if not os.path.isdir(v):
            raise ValueError("The path is not a directory")
        if not os.access(v, os.R_OK):
            raise ValueError("The directory is not accessible")
        return v

class Retrieve(BaseModel):
    query: str


@app.get("/", tags=["check"])
def home():
    return {"Health": "OK"}


@app.post("/ingest_data")
def data_ingestion(payload: IngestData):
    # log received path
    num_docs, num_indexed = pipeline(payload.path)
    return {
        "message": f"Read {num_docs} Document from {payload.path}. Indexed {num_indexed} Chunks into DB"
    }

@app.post("/retrieve")
def retrieve(payload: Retrieve):
    # log received query
    rertieved_context = get_context(payload.query)
    return rertieved_context

def get_llm_response(input):
    reqUrl = config_ns.llm['url']

    headersList = {"Accept": "*/*", "Content-Type": "application/json"}

    payload = json.dumps(
        {
            "text_input": input,
            "max_tokens": config_ns.llm['max_tokens'],
            "bad_words": config_ns.llm['bad_words'],
            "stop_words": config_ns.llm['stop_words'],
            "top_k": config_ns.llm['top_k'],
            "top_p": config_ns.llm['top_p'],
            "temperature": config_ns.llm['temperature'],
            "random_seed": config_ns.llm['random_seed'],
            "length_penalty": config_ns.llm['length_penalty'] 
        }
    )

    response = requests.request("POST", reqUrl, data=payload, headers=headersList, stream=False)
    return response


@app.post("/llm")
def call_llm(payload: Retrieve):
    # call retrieve
    context = retrieve(payload)
    # construct prompt using langchain
    input = prompt.format(context=context, question=payload.query)
    # call llm
    output = get_llm_response(input).json()
    print(output['text_output'])
    parsed_output = output["text_output"].split('bot_response:')[-1].split('</s><s>')[0].lstrip('"')
    return {"text_output": parsed_output}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999, reload=True)
