from llama_cpp import Llama

llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf")
output = llm("What is retrieval augmented generation?", max_tokens=100)
print(output["choices"][0]["text"])
