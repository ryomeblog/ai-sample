from llama_cpp import Llama

def generate_response(llm, prompt):
    print("Loading...")
    result = llm(prompt, max_tokens=1000, echo=True,)
    response = result['choices'][0]['text']
    print(f"Llama2: {response}")

llm = Llama(model_path="../data/codellama-13b-instruct.ggmlv3.Q8_0.bin")

while True:
    user_input = input("You: ")
    if user_input == "quit":
        break
    generate_response(llm, user_input)
