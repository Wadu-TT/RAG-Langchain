#RAG-Langchain
Files and Functionality
3_chain_concept.py

This script demonstrates the use of Langchain to create a sequence of tasks. It includes:

    LLMChain and SequentialChain: Used to create a series of language model tasks.
    PromptTemplate: Templates for generating prompts for the language models.
    Location Chain: Generates a classic dish based on a location.
    Dish Chain: Provides a recipe for the dish.
    Recipe Chain: Estimates the cooking time for the dish.

agents.py

This script showcases the use of Langchain agents to interact with a pandas DataFrame:

    Pandas DataFrame Agent: An agent to perform operations on a DataFrame loaded from an example.csv file.
    Agent Invocation: Example of invoking the agent to query the number of rows in the DataFrame.

companyPolicies.txt

This file contains the company's policies, including:

    Code of Conduct
    Recruitment Policy
    Internet and Email Policy
    Mobile Phone Policy
    Smoking Policy
    Drug and Alcohol Policy
    Health and Safety Policy
    Anti-discrimination and Harassment Policy
    Discipline and Termination Policy

langchain.py

This script includes various examples and utilities for working with Langchain and IBM Watson models:

    Model Setup: Installing required packages and setting up IBM Watson and OpenAI models.
    Prompt Templates: Creating and invoking prompts for different tasks.
    Document Loading and Processing: Loading and processing documents using Langchain's document loaders.
    Retrieval and QA: Setting up retrieval QA using Langchain and IBM Watson models.

reg_policy_qa.py

This script provides a question-answering system for the company's policies:

    Data Loading: Loading and splitting the text from the companyPolicies.txt file.
    Context Encoding and Indexing: Encoding the contexts using DPR encoders and creating a FAISS index.
    Search and Answer Generation: Searching relevant contexts and generating answers using GPT-2.

Usage

To run the scripts, you can use the following commands:
sh

python 3_chain_concept.py
python agents.py
python langchain.py
python reg_policy_qa.py

Contributing

If you would like to contribute to the project, please fork the repository and create a pull request with your changes.
