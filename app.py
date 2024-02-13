from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from main import AsyncCallbackHandler, create_gen, prompt_ai
# from langcorn import create_service
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

class Input(BaseModel):
    prompt: str

class Query(BaseModel):
    text: str

class Output(BaseModel):
    result: str



app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/prompt")
async def input(query: Query):
    # stream_it = AsyncCallbackHandler()
    # gen = create_gen(query.text, stream_it)
    # return StreamingResponse(gen, media_type="text/event-stream")
    res = prompt_ai(query.text)
    output = Output(result=res)
    return output


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":

    import uvicorn

    uvicorn.run("app:app", reload=True, port=3000)
