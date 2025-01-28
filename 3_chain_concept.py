from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.llms import IBMWatsonxAI
from langchain.prompts import PromptTemplate

template = """Your job is to come up with a classic dish from the area that the user suggests.
{location}
YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template=template,input_variable = ['location'])

# Define the LLM
mixtral_llm = OpenAI(model_name="text-davinci-003")

#chain 1
watsonx_llm = IBMWatsonxAI(model_name="ibm-watsonx-ai")
location_chain = LLMChain(llm=watsonx_llm, prompt=prompt_template, output_key="meal")

template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.
YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template=template,input_variable = ['meal'])
#chain2
dish_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key="recipe")
template = """Given the recipe {recipe}, estimate how much time i need to cook it
YOUR RESPONSE:
"""

prompt_template = PromptTemplate(template=template,input_variable = ['recipe'])
#chain 3
recipe_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template, output_key="time")
#OVERALL CHAIN
overall_chain = SequentialChain(chains=[location_chain, dish_chain, recipe_chain],
input_variables = ['location'],
output_variables = ['meal','recipe','time'],
verbose =   True )  
overall_chain.run(location = "China")