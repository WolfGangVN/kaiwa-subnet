from concurrent.futures import ThreadPoolExecutor
import os
import time
import fastapi
from fastapi import FastAPI
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from kaiwa_subnet.base.config import KaiwaBaseSettings
from kaiwa_subnet.base.infer import InferenceEngine
from cachetools import TTLCache
import mysql.connector
import logging

logging.basicConfig(level=logging.DEBUG)
import os
from dotenv import load_dotenv
load_dotenv()
from asyncio import Semaphore
app = FastAPI()
lock = Semaphore(1)

from pydantic import BaseModel
model_name = "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit"
d = InferenceEngine(settings=KaiwaBaseSettings(model=model_name))
db = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
)

cache = TTLCache(maxsize=2000, ttl=500)
class InferRequest(BaseModel):
    req: str
    
def hash_string(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()

def check_exists(hashkey: str) -> bool:
    cursor = db.cursor()
    cursor.execute(f"SELECT * FROM quora_questions.questions WHERE hash = '{hashkey}'")
    result =  cursor.fetchall()
    if result and len(result) > 0:
        return True, result[0][3]
    return False, None

def update_response(hashkey: str, response: str) -> None:
    # cursor = db.cursor()
    # cursor.execute(f"UPDATE quora_questions.questions SET response = '{response}' WHERE hash = '{hashkey}'")
    # db.commit()
    # Update response in db with escape string for response
    cursor = db.cursor()
    cursor.execute(f"UPDATE quora_questions.questions SET response = %s WHERE hash = %s", (response, hashkey))
    db.commit()
    
@app.post("/infer")
async def infer(infer_request: InferRequest):
    hashkey = hash_string(infer_request.req)
    # print(f"Hashkey: {hashkey}")
    exists, resp = check_exists(hashkey)
    # print(f"Exists: {exists} Response: {resp}")
    if exists and resp:
        logging.info(f"Hashkey: {hashkey} is in db, returning response")
        return {
            "choices": [
                {
                    "message": {
                        "content": resp,
                        "role": "assistant",
                    }
                }
            ]
        }
    
    if hashkey in cache.keys():
        logging.info(f"Hashkey: {hashkey} is in cache, waiting for response")
        # pooling 10 seconds to get response
        start = time.time()
        while True:
            if cache[hashkey] is not None:
                logging.info(f"Got response for hashkey: {hashkey} with response: {cache[hashkey]['choices'][0]['message']['content']}")
                return cache[hashkey]
            if time.time() - start > 20:
                break
            time.sleep(0.1)
        
        logging.info(f"Cannot get response for hashkey: {hashkey}")
        return {
            "choices": [
                {
                    "message": {
                        "content": "Cannot get response",
                        "role": "assistant",
                    }
                }
            ]
        }
    
    logging.info(f"Hashkey: {hashkey} is not in cache, sending request to model")
    cache[hashkey] = None
    # lock
    async with lock:
        res = await d.chat(
            input=ChatCompletionRequest(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": infer_request.req,
                    }
                ],
            )
        )
    cache[hashkey] = res
    update_response(hashkey, res["choices"][0]["message"]["content"])    
    logging.info(f"Updated response for hashkey: {hashkey} with response: {res['choices'][0]['message']['content']}")
    return res
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
