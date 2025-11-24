import openai
from SPARQLWrapper import SPARQLWrapper, CSV, SPARQLExceptions, JSON
from dotenv import load_dotenv
import os
import io

#loads env file holding sensitive information like api keys and passwords
load_dotenv()


#---------------------------
# ChatGPT Functions
# 

def gpt_call(system,prompt,temperature):
    """
    This fuction calls gpt api and returns the answer
    Args:
        system (string): instructions for the LLM before the query
        prompt (string): chatgpt prompt
        temperature (flat): randomness of output
    Returns:
        str: chatgpt response
    """
    client = openai.OpenAI(api_key=os.getenv("gpt_api_key"))
    llm_model = os.getenv("llm_model","gpt-4o")
    tokens=int(os.getenv("max_tokens",10000))
    response = client.chat.completions.create(
    model=llm_model,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ],
        max_tokens=tokens,
        temperature=temperature
    )
    return response.choices[0].message.content

def remove_first_and_last_line(text):
    """
    This removes the first and last line of a piece of text(used because chatgpt output has ``` brackets around them)
    Args:
        text (string): a piece of text
    Returns:
        str: chatgpt response (returns only the contructed query)
    """
    lines = text.splitlines()
    if len(lines) <= 2:
        return ""  # Nothing to return if there's only 1 or 2 lines
    return "\n".join(lines[1:-1])

#---------------------------
# File Functions
# 

def load_template(filename):
    """
    This fuction opens a file and reads it
    Args:
        filename (string): path to the file name *if not in folder have to put the full path* 
    Returns:
        str: everything inside of the file
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


#---------------------------
# GraphDB Functions
# 

def query_database(query):
    """
    This function queries the knowledge graph database (graphDB) and writes the output to query-result.csv
    Args:
        query (string): a sparql query for the database
    Returns:
        None : writes output to query-result.csv
    """
    sparql = SPARQLWrapper(os.getenv("graphdb_repo"))
    sparql.setCredentials(os.getenv("graphdb_username"), os.getenv("graphdb_password"))
    sparql.setReturnFormat(CSV)
    try:
        sparql.setQuery(query)
        response = sparql.query()
        csv_results = response.convert().decode("utf-8")
        return io.StringIO(csv_results)
    except SPARQLExceptions.QueryBadFormed as e:
        print("SPARQL query is malformed:", e)
        return "malformed query error"
    except Exception as e:
        print("An unexpected error occurred:", e)
        return "unexpected error"

