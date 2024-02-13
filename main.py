from typing import Any
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models.cohere import ChatCohere
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.schema import LLMResult
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
currentdir = os.path.dirname(__file__)
# Load a PDF document and split it into sections
loader = PyPDFLoader(os.path.join(currentdir, "./book.pdf"))
docs = loader.load_and_split()
cohere_apikey = os.environ.get("COHERE_API_KEY")

os.environ["COHERE_API_KEY"] = cohere_apikey
# Initialize the OpenAI chat model
llm = ChatCohere(temperature=0.8)
# llm = ChatCohere(temperature=0.8, streaming=True, callbacks=[])

# Initialize the OpenAI embeddings
embeddings = CohereEmbeddings()


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        print(token)
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""



# async def prompt_ai(prompt: str, stream_it: AsyncCallbackHandler):
def prompt_ai(prompt: str):
    # Load the Chroma database from disk
    chroma_db = Chroma(
        persist_directory="data",
        embedding_function=embeddings,
        collection_name="lc_chroma_demo",
    )

    # Get the collection from the Chroma database
    collection = chroma_db.get()

    # If the collection is empty, create a new one
    if len(collection["ids"]) == 0:
        # Create a new Chroma database from the documents
        chroma_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="data",
            collection_name="lc_chroma_demo",
        )

        # Save the Chroma database to disk
        chroma_db.persist()

    # Prepare query
    query = prompt
    # query = "What is this document about?"

    # print('Similarity search:')
    # print(chroma_db.similarity_search(query))

    # print('Similarity search with score:')
    # print(chroma_db.similarity_search_with_score(query))

    # # Add a custom metadata tag to the first document
    # docs[0].metadata = {
    #     "tag": "demo",
    # }

    # # Update the document in the collection
    # chroma_db.update_document(
    #     document=docs[0],
    #     document_id=collection['ids'][0]
    # )

    # Find the document with the custom metadata tag
    # collection = chroma_db.get()
    # collection = chroma_db.get(where={"tag" : "demo"})
    # llm.callbacks = [stream_it]
    # Prompt the model
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=chroma_db.as_retriever()
    )
    # Execute the chain
    response = chain(query)
    return response["result"]
    # await chain.ainvoke(input={"query": query})


async def create_gen(query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(prompt_ai(query, stream_it))
    async for token in stream_it.aiter():
        print("token", token)
        yield token
    await task

