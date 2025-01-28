import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

df = pd.read_csv(
    "example.csv"
)
agent = create_pandas_dataframe_agent (
    mixtral_llm = OpenAI(model_name="text-davinci-003"),
    df = df,
    verbose = True,
    return_intermediate_steps=True

)
agent.invoke("how many rows in the dataframe?")