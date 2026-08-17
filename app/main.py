from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.sse import EventSourceResponse, ServerSentEvent

from base64 import b64encode
from pydantic import BaseModel
from Cover.lib.steganography import *
from Cover.lib.mec_math import pad_with_rand


app = FastAPI()


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
TOPK = 40
BLOCK_SIZE = 8
NUM_BLOCKS = 64

steg = Steganographer(MODEL_NAME, topk=TOPK, block_size=BLOCK_SIZE)


class EncModel(BaseModel):
    context: str
    message: str



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
# 1234


@app.post("/encode" , response_class=EventSourceResponse)
async def encode_message_stream(data: EncModel):
    padded_ct = pad_with_rand(data.message.encode(), BLOCK_SIZE, NUM_BLOCKS)
    assert len(padded_ct) == NUM_BLOCKS
    encodings = steg.encode(padded_ct, data.context)
    for word, _1, _2 in encodings: # ignore istar and delta_mu
        yield ServerSentEvent(raw_data=word, event='token')
    yield ServerSentEvent(data="[DONE]", event="done")


@app.post("/decode", response_class=EventSourceResponse)
async def decode_message_stream(data: EncModel):
    tokens = steg.llm_sampler.tokenizer.encode(data.message)
    decodings = steg.decode(tokens, data.context, NUM_BLOCKS)
    for pred_barr in decodings:
        out_str = b''.join(pred_barr).decode('latin-1')
        yield ServerSentEvent(raw_data=out_str, event='token')
    yield ServerSentEvent(data="[DONE]", event="done")



# static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")