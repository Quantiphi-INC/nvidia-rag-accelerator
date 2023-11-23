import requests
import json
import os

BACKEND_LLM_ENDPOINT=os.getenv("BACKEND_LLM_ENDPOINT")

def query_llm(input):
    reqUrl = BACKEND_LLM_ENDPOINT

    headersList = {
    "Accept": "*/*",
    "Content-Type": "application/json" 
    }

    payload = json.dumps({
    "query":input
    })

    response = requests.request("POST", reqUrl, data=payload,  headers=headersList)
    return response