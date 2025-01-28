pip install "ibm-watsonx-ai==1.0.4"
pip install "ibm-watson-machine-learning==1.0.357"
pip install "langchain==0.2.1" 
pip install "langchain-ibm==0.1.7"
pip install "langchain-community==0.2.1"
pip install "langchain-experimental==0.0.59"
pip install "langchainhub==0.1.17"
pip install "pypdf==4.2.0"
pip install "chromadb"

# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

model_id = 'mistralai/mixtral-8x7b-instruct-v01'

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = "skills-network"

model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
msg = model.generate("In today's sales meeting, we ")
print(msg['results'][0]['generated_text'])
mixtral_llm = WatsonxLLM(model = model)
print(mixtral_llm.invoke("Who is man's best friend?"))
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
msg = mixtral_llm.invoke(
    [
        SystemMessage(content="You are a helpful AI bot that assists a user in choosing the perfect book to read in one short sentence"),
        HumanMessage(content="I enjoy mystery novels, what should I read?")
    ]
)
print(msg)
msg = mixtral_llm.invoke(
    [
        SystemMessage(content="You are a supportive AI bot that suggests fitness activities to a user in one short sentence"),
        HumanMessage(content="I like high-intensity workouts, what should I do?"),
        AIMessage(content="You should try a CrossFit class"),
        HumanMessage(content="How often should I attend?")
    ]
)
print(msg)
msg = mixtral_llm.invoke(
    [
        HumanMessage(content="What month follows June?")
    ]
)
print(msg)
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
input_ = {"adjective": "funny", "topic": "cats"}  # create a dictionary to store the corresponding input to placeholders in prompt template
prompt.invoke(input_)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

input_ = {"topic": "cats"}

prompt.invoke(input_)
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}

prompt.invoke(input_)
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,  # The maximum length that the formatted examples should be.
)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    # And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
output_parser = JsonOutputParser(pydantic_object=Joke)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | mixtral_llm | output_parser

chain.invoke({"query": joke_query})
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | mixtral_llm | output_parser
chain.invoke({"subject": "ice cream flavors"})
from langchain_core.documents import Document
Document(page_content="""Python is an interpreted high-level general-purpose programming language. 
                        Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "About Python",
             'my_document_create_time' : 1680013019
         }) 
Document(page_content="""Python is an interpreted high-level general-purpose programming language. 
                        Python's design philosophy emphasizes code readability with its notable use of significant indentation.""")
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf")
document = loader.load()
document[2]  # take a look at the page 2
print(document[1].page_content[:1000])  # print the page 1's first 1000 tokens
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
web_data = loader.load()

print(web_data[0].page_content[:1000])
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")  # define chunk_size which is length of characters, and also separator.
chunks = text_splitter.split_documents(document)
print(len(chunks))
print(len(chunks))
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
from langchain_ibm import WatsonxEmbeddings

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)
texts = [text.page_content for text in chunks]

embedding_result = watsonx_embedding.embed_documents(texts)
embedding_result[0][:5]
from langchain.vectorstores import Chroma
docsearch = Chroma.from_documents(chunks, watsonx_embedding)
query = "Langchain"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
retriever = docsearch.as_retriever()
docs = retriever.invoke("Langchain")
docs[0]
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
# Set two splitters. One is with big chunk size (parent) and one is with small chunk size (child)
parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator='\n')

vectorstore = Chroma(
    collection_name="split_parents", embedding_function=watsonx_embedding
)

# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(document)
len(list(store.yield_keys()))
sub_docs = vectorstore.similarity_search("Langchain")
print(sub_docs[0].page_content)
retrieved_docs = retriever.invoke("Langchain")
print(retrieved_docs[0].page_content)
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=mixtral_llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "what is this paper discussing?"
qa.invoke(query)

from langchain.memory import ChatMessageHistory
chat = mixtral_llm

history = ChatMessageHistory()

history.add_ai_message("hi!")

history.add_user_message("what is the capital of France?")
history.messages
ai_response = chat.invoke(history.messages)
ai_response
history.add_ai_message(ai_response)
history.messages
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
conversation = ConversationChain(
    llm=mixtral_llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
conversation.invoke(input="Hello, I am a little cat. Who are you?")
conversation.invoke(input="What can you do?")
conversation.invoke(input="Who am I?.")
from langchain.chains import LLMChain
template = """Your job is to come up with a classic dish from the area that the users suggests.
                {location}
                
                YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template=template, input_variables=['location'])

# chain 1
location_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='meal')
location_chain.invoke(input={'location':'China'})
from langchain.chains import SequentialChain
template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.

                YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template=template, input_variables=['meal'])

# chain 2
dish_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='recipe')
template = """Given the recipe {recipe}, estimate how much time I need to cook it.

                YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template=template, input_variables=['recipe'])

# chain 3
recipe_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key='time')
# overall chain
overall_chain = SequentialChain(chains=[location_chain, dish_chain, recipe_chain],
                                      input_variables=['location'],
                                      output_variables=['meal', 'recipe', 'time'],
                                      verbose= True)
from pprint import pprint
pprint(overall_chain.invoke(input={'location':'China'}))
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm=mixtral_llm, chain_type="stuff", verbose=False)
response = chain.invoke(web_data)
print(response['output_text'])
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
python_repl = PythonREPL()
python_repl.run("a = 3; b = 1; print(a+b)")
from langchain_experimental.tools import PythonREPLTool
tools = [PythonREPLTool()]
from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor
instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""

# here you will use the prompt directly from the langchain hub
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)
agent = create_react_agent(mixtral_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # tools were defined in the toolkit part above
agent_executor.invoke(input = {"input": "What is the 3rd fibonacci number?"})