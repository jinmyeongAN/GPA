
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable

elastic = RemoteRunnable("http://localhost:8000/rag-elasticsearch")
chroma = RemoteRunnable("http://localhost:8000/rag-chroma")

user_id = "username_filename"

lecture_slide_str="""
Page 18
Stochastic gradient descent (SGD)
- SGD: Most widely used algorithm for deep learning.
	- Do not confuse with (deterministic) gradient descent (GD).
	- SGD uses minibatches with some important modifications.
		- Changes in learning rate!

Page 19
Stochastic gradient descent (SGD)
- Learning rate 𝜖 (cid:3038) necessary to gradually decrease.
	- Minibatches introduce noise that do not disappear along even at the minimum.
- Common strategy: decay learning rate until iteration 𝜏 .
	- Set 𝛼 (cid:3404) (cid:3038) (cid:3099) and leave 𝜖 (cid:3038) constant after iteration 𝜏 .
		- 𝜏 should allow a few hundred passes through the training set.
		- 𝜖 (cid:3099) should be roughly 1% of 𝜖 (cid:2868) .
		- 𝜖 (cid:2868) is set higher than the best value for the first 100 iterations
		(Monitor the initial results and use a higher value, not too high.)
	- There exist many other schemes for learning rates.

Page 20
Learning rate
- Use a learning rate decay over time (step decay)
	- E.g. decay learning rate by half every few epochs.
	- Other popular choices:
	- Adaptive methods, such as SGD, SGD+Momentum, Adagrad, RMSProp, Adam, all have learning rate as a hyperparameter. (will be explained later).

Page 21
Learning rate
- Coarse‐to‐fine cross‐validation
	- First stage: only a few epochs to get rough idea of what params work.
	- Second stage: longer running time, finer search
	- … (repeat as necessary)
		Random Search for Hyper-Parameter Optimization Bergstra and Bengio, 2012
"""

#print(lecture_slide_str)

#response = elastic.invoke({ "context": lecture_slide_str})
#print(response)


response = chroma.invoke({ "context": lecture_slide_str})
print(response)


#response = elastic.stream({ "context": lecture_slide_str})
#print(response)

#for chunk in response:
#    print(chunk, end="", flush=True)


#elastic.invoke({"topic": "parrots"})

"""
# or async
await joke_chain.ainvoke({"topic": "parrots"})

prompt = [
    SystemMessage(content='Act like either a cat or a parrot.'),
    HumanMessage(content='Hello!')
]

# Supports astream
async for msg in anthropic.astream(prompt):
    print(msg, end="", flush=True)

prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me a long story about {topic}")]
)

# Can define custom chains
chain = prompt | RunnableMap({
    "openai": openai,
    "anthropic": anthropic,
})

chain.batch([{ "topic": "parrots" }, { "topic": "cats" }])
"""