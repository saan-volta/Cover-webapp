from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.sse import EventSourceResponse, ServerSentEvent

from pydantic import BaseModel
from Cover.lib.steganography import *
from Cover.lib.mec_math import pad_with_rand, cycle_k

import random


app = FastAPI()


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
TOPK = 40
BLOCK_SIZE = 8
KEY_SIZE = 11
NUM_BLOCKS = 64
assert NUM_BLOCKS*BLOCK_SIZE % 8 == 0

steg = Steganographer(MODEL_NAME, topk=TOPK, block_size=BLOCK_SIZE)


class EncModelNoKey(BaseModel):
    context: str
    message: str

class EncModelWithKey(BaseModel):
    context: str
    message: str
    key: int


# @app.post("/encode")
# async def encode_message(data: EncModel):
#     padded_ct = pad_to_maxlen(data.message)
#     assert len(padded_ct) == MAX_NUM_BLOCKS
#     _, final_text = steg.encode(padded_ct, data.context)
#     return {"result" : final_text}


# @app.post("/decode")
# async def decode_message(data: EncModel):
#     tokens = steg.llm_sampler.tokenizer.encode(data.message)
#     ct = steg.decode(tokens, data.context, MAX_NUM_BLOCKS)
#     return {"result" : ct.decode()}
# 6


@app.post("/encode" , response_class=EventSourceResponse)
async def encode_message_stream(data: EncModelNoKey):
    message = data.message.encode()
    key: int = random.getrandbits(KEY_SIZE)
    cycled_key = cycle_k(key, len(message), KEY_SIZE)
    ciphertext = bytes_xor(message, cycled_key)

    padded_ct = pad_with_rand(ciphertext, BLOCK_SIZE, NUM_BLOCKS)
    assert len(padded_ct) == NUM_BLOCKS
    encodings = steg.encode(padded_ct, data.context)
    yield ServerSentEvent(data=key, event='key')
    for word, _star_idx, _delta_H, d_KL in encodings: # ignore istar and delta_mu
        yield ServerSentEvent(raw_data=word, event='token')
    yield ServerSentEvent(data="[DONE]", event="done")


@app.post("/decode", response_class=EventSourceResponse)
async def decode_message_stream(data: EncModelWithKey):
    tokens = steg.llm_sampler.tokenizer.encode(data.message)
    cycled_key_full_length = cycle_k(data.key, NUM_BLOCKS*BLOCK_SIZE//8, KEY_SIZE)
    decodings = steg.decode(tokens, data.context, NUM_BLOCKS)
    for pred_bitarr_list in decodings:
        out_str = bytes_xor( b''.join(pred_bitarr_list) , cycled_key_full_length).decode('latin-1')
        yield ServerSentEvent(raw_data=out_str, event='token')
    yield ServerSentEvent(data="[DONE]", event="done")



# static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")