from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model=OllamaLLM(model="llama3.2") #model="llama3.2" is the default

# Set up the prompt template
template = """"   You are an expert in answering questions related to a pizza restaurent.
here are some reviews: {reviews}
here are some questions: {questions}"""

# Create a ChatPromptTemplate instance
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model # Create a chain with the prompt and model


while True:
    print("\n\n----------------------------------------")
    question=input("Enter your question (or 'exit' to quit): ")
    print("\n\n----------------------------------------")
    if question.lower() == 'exit':
        break

    # Use the retriever to get relevant reviews
    reviews = retriever.invoke(question)

    # Define the input data
    result = chain.invoke({"reviews": reviews, "questions": question})

    # Print the result
    print(result)