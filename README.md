# GPA

## Quick start

**Install torch** (you SHOULD consider your CUDA version)

```$ pip install torch torchvision torchaudio```

**Install requirements**

```$ pip install -r requirements.txt```

**Install linter**

```$ make style```

**Set your openAI api key**

1. Make `const.py` file in `GPA/packages/rag-chroma/rag_chroma/src/config`` directory
2. fill in the file like followings
```python
OPENAI_API_KEY = {YOUR_API_KEY}
```

**Execute RAG**

1. Go to the `main.py` file in `GPA/packages/rag-chroma/rag_chroma/main.py`
2. `$ python main.py`