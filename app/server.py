
import sys
sys.path.append('/home/jhkim980112/workspace/code/Deep_Learning_Proj/GPA/packages/rag-elasticsearch')
sys.path.append('/home/jhkim980112/workspace/code/Deep_Learning_Proj/GPA/packages/rag-chroma')


# import path 확인
#for i in sys.path:
#    print(i)

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_chroma import chain as rag_chroma_chain
from rag_elasticsearch import chain as rag_elasticsearch_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chroma_chain, path="/rag-chroma")

add_routes(app, rag_elasticsearch_chain, path="/rag-elasticsearch")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
