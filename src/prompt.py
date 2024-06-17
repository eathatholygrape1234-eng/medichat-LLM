from langchain_core.prompts import ChatPromptTemplate

# Define the system prompt to guide the model's behavior
system_prompt = (
    "You are a medical professional and act like one. Use the following pieces of information to answer the user's question accurately. "
    "If you don't know the answer, just say that you don't know. Do not add any details by yourself. "
    "Context: {context}"
    "Only return the helpful answer below and nothing else. "
    "Helpful answer:"
)

# Create the prompt template with the system and human message structure
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
hello = "Hello, I am a medical professional and I will help you with your question. Please provide me with the context of your question."
