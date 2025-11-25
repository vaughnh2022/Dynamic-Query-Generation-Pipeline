#imports
import csv
from dotenv import load_dotenv
import pandas as pd
from collections import deque
import pandas as pd
from api_commands import gpt_call, remove_first_and_last_line, load_template, query_database

#   Main Function
def grag_pipeline(question,temperature):
    """
    This function runs the complete pipeline once
    Note it will output all information into the terminal
    Args:
        question (string): user inputted question
    Returns:
        str: markdown nl output
        str: sparql query
    """
    if question=="test":
        pass
        #return nl_output('test.csv'), "test query"
    print("\n\n\n\n\n\n--------------------------------------------\n\t\t\tSTARTING PIPELINE\n\nquestion is:\n",question,"\n\n")
    classes=llm_output_to_list(llm_pick_classes(question,temperature)) #runs through LLM to select what classes are important given the question
    print("classes picked are:\n",classes,"\n\n")
    class_path=get_class_path(classes)  #finds the class path using bfs(breadth-first search)
    print("class path is:\n",class_path,"\n\n")
    all_prop=get_class_properties(classes) #collects all properties for each of the selected classes
    selected_prop=get_selection(all_prop,question,temperature)  #runs through LLM to select what properties are important
    print("choosen properties are:\n",selected_prop,"\n\n")
    example_query=create_example_query(class_path,convert_section_to_question(selected_prop),convert_section_to_optional(selected_prop)) #creates the dynamic example query
    print("example query is:\n",example_query,"\n\n")
    sparql_query=create_sparql_query(example_query,question,temperature) #runs this example query through the LLM again with query construction rules and outputs the final query
    print("sparql query is:\n",sparql_query,"\n\n")
    print("querying database\n\n")
    answer=query_database(sparql_query)  #queries the database with the given query
    print("complete output is\n",answer,"\n\n\n")
    answer_list=column_to_list(answer)  #extracts the first collumn of the csv file output for metric running
    print("output is:\n",answer_list)
    return (classes,selected_prop,sparql_query,answer_list) #returns the answer_list, selected classes, selected properties, and query to be added to metric output

#extract important classes
def llm_pick_classes(question,temperature):
    """
    This function defines the system for the llm to select classes
    Note the prompt is just the question itself
    Args:
        question (string): user inputted question
    Returns:
        str: string output of all classes selected
    """
    prompt = """You are selecting relevant classes given a question over electronic health record and ontology data, the classes are listed below:

Condition
    - For questions with diseases use this template
Procedure
icd9
    - For any questions with Condition or Procedure being a relevant class also include this class
icd10CM
    - For any questions with Condition being a relevant class also include this class
icd10PCS
    - For any questions with Producedure being a relevant class also include this class
Encounter
    - Select this if any specific dates are mentioned in the question
Location
    - Select this if any specific sections of the hospital are in the question
LocationEncounter
    - Select this class when either Location or Encounter are in the question
Organization
    - Use this if the hospital is ever in the question
Patient
    - If given a patient id or any question with patients use this 
Specimen

---

### Instructions:
Given a question, **return only** the most relevant classes **in this exact format (no other text):**

Condition,Specimen,Patient
"""
    return gpt_call(prompt,question,temperature)

def llm_output_to_list(classes):
    """
    This function converts the string output of the llm to a list
    This is used to covert the classes to a python list for easier interpretation
    Args:
        classes (string): string listing all collected classes
    Returns:
        list (str): all classes selected 
    """
    word=""
    class_list=[]
    for x in classes:
        if x==' ':
            continue
        if x==',':
            class_list.append(word)
            word=""
        else:
            word=word+x
    class_list.append(word)
    return class_list

#gets the path between all the selected classes for the query template
def get_class_path(classes):
    """
    This function uses bps(breadth-first search) to find a path that visits all possible classes to output into the example query
    This makes sure there is no issue in connecting classes while querying the data
    Args:
        classes (string): a string of all selected classes
    Returns:
        str: sparql code of all classes; looks something like this ?specimen a flat:Specimen .
    """
    if len(classes) == 1:
        return f"?{classes[0].lower()} a flat:{classes[0]} .\n"

    # Graph connectivity (undirected for path search)
    graph = {
        "Location": ["Organization", "LocationEncounter", "Encounter"],
        "LocationEncounter": ["Location", "Encounter"],
        "Encounter": ["LocationEncounter", "Organization", "Patient", "Procedure", "Condition"],
        "Organization": ["Location", "Encounter", "Patient"],
        "Procedure": ["Encounter", "icd9", "icd10PCS"],
        "Condition": ["Encounter", "Patient", "icd9", "icd10CM"],
        "Patient": ["Encounter", "Condition", "Organization","Specimen"],
        "icd9": ["Condition", "Procedure"],
        "icd10CM": ["Condition"],
        "icd10PCS": ["Procedure"],
        "Specimen": ["Patient"]
    }

    # Directed version (ontology reference direction)
    directional_graph = {
        "Location": ["Organization"],
        "LocationEncounter": ["Location"],
        "Encounter": ["LocationEncounter", "Organization", "Patient"],
        "Organization": [],
        "Procedure": ["Encounter", "icd9", "icd10PCS"],
        "Condition": ["Encounter", "Patient", "icd9", "icd10CM"],
        "Patient": ["Organization"],
        "icd9": [],
        "icd10CM": [],
        "icd10PCS": [],
        "Specimen":["Patient"]
    }

    # BFS shortest path
    def bfs_shortest_path(start, goal):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == goal:
                return path
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # Build SPARQL pattern using a spanning tree approach
    ans = ""
    declared = set()
    declared_rels = set()
    connected = set()
    
    # Start with first class
    first_class = classes[0]
    ans += f"\t?{first_class.lower()} a flat:{first_class} .\n"
    declared.add(first_class)
    connected.add(first_class)
    
    # Connect remaining classes one by one to the growing tree
    remaining = list(classes[1:])
    
    while remaining:
        best_connection = None
        best_distance = float('inf')
        
        # Find the closest unconnected class to any connected class
        for target in remaining:
            for source in connected:
                path = bfs_shortest_path(source, target)
                if path and len(path) - 1 < best_distance:
                    best_distance = len(path) - 1
                    best_connection = (source, target, path)
        
        if not best_connection:
            break
            
        source, target, path = best_connection
        
        # Add all nodes in the path
        for i in range(len(path)):
            node = path[i]
            if node not in declared:
                ans += f"\t?{node.lower()} a flat:{node} .\n"
                declared.add(node)
            connected.add(node)
        
        # Add relationships along the path
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            
            # Choose direction based on ontology structure
            if node2 in directional_graph[node1]:
                rel = f"{node1.lower()} flat:{node2.lower()}_reference ?{node2.lower()} ."
            elif node1 in directional_graph[node2]:
                rel = f"{node2.lower()} flat:{node1.lower()}_reference ?{node1.lower()} ."
            else:
                # fallback
                rel = f"{node1.lower()} flat:{node2.lower()}_reference ?{node2.lower()} ."
            
            if rel not in declared_rels:
                ans += f"\t?{rel}\n"
                declared_rels.add(rel)
        
        remaining.remove(target)
    
    return ans

#  get all properties for a given classes
def get_class_properties(classes):
    """
    This function holds a dictionary of all properties and their related class, and returns the class and properties given a list of classes
    Args:
        classes (string): llm selected relevant classes
    Returns:
        str: a list of classes and all of their properties, in this format
            class1[property1,property2]
            class2[property3,property4]
    """
    hashmap = {
        "Patient": "[flat:deceased_datetime, flat:gender, flat:age, flat:race, flat:ethnicity, flat:organization_reference, flat:id, flat:identifier]",
        "Condition": "[flat:encounter_reference, flat:patient_reference, flat:icd9_reference, flat:id, flat:identifier, flat:category_code, flat:icd10cm_reference]",
        "LocationEncounter": "[flat:location_reference, flat:period_start, flat:period_end, flat:patient_reference]",
        "Encounter": "[flat:patient_reference, flat:encounter_reference, flat:organization_reference, flat:admission_class_code, flat:services_code, flat:admission_type, flat:identifier_use, flat:admit_source_code, flat:discharge_disposition_code, flat:id, flat:identifier, flat:status, flat:period_start, flat:period_end, flat:snomed_display, flat:snomed_code, flat:hcpcs_display, flat:hcpcs_code]",
        "Location": "[flat:organization_reference, flat:name, flat:id, flat:location_physical_type_code, flat:location_physical_type_display, flat:status]",
        "Procedure": "[flat:encounter_reference, flat:patient_reference, flat:id, flat:identifier, flat:status, flat:period_start, flat:period_end, flat:d_items_code, flat:d_items_display, flat:bodysite_code, flat:category_code, flat:icd9_reference, flat:performed_datetime, flat:icd10pcs_reference]",
        "Organization": "[flat:active, flat:name, flat:id, flat:organization_type_code, flat:organization_type_display, flat:identifier]",
        "icd9": "[umls:cui, umls:tui, rdfs:subClassOf, skos:notation,skos:prefLabel]",
        "icd10CM": "[rdfs:subClassOf, skos:prefLabel, skos:notation, skos:altLabel, icd10CM:ORDER_NO, umls:cui, umls:tui, umls:hasSTY, icd10CM:CODE_FIRST, icd10CM:EXCLUDES2, icd10CM:EXCLUDES1, icd10CM:USE_ADDITIONAL, icd10CM:CODE_ALSO, icd10CM:NOTE]",
        "icd10PCS": "[rdfs:subClassOf, skos:prefLabel, skos:notation, skos:altLabel, umls:cui, umls:tui, umls:hasSTY, icd10PCS:ORDER_NO, icd10PCS:ADDED_MEANING]",
        "Specimen": "[flat:collected_datetime, flat:id, flat:identifier, flat:lab_fluid_code, flat:patient_reference, flat:spec_type_desc_code, flat:spec_type_desc_display]"
    }
    ans=""
    for x in classes:
        ans=ans+x+hashmap[x]+"\n"
    return ans

#llm selects relevant properties
def get_selection(classes_list,question,temperature):
    """
    This function prompts the llm to select which properties of the selected classes are relevant 
    Args:
        classes (string): llm selected relevant classes
        question (string): user inputted question
    Returns:
        str: a list of classes and all of their selected relevant properties, in this format
            class1[property1,property2]
            class2[property3,property4]
    """
    call = gpt_call(
    "Given a question you are to select all relevant properties in the knowledge graph. Always select ids, codes, and displays. Do not select references.",
    f"""
    Here is the classes and property list in this structure:

    class1[property1,property2,property3]
    class2[property1,property2]

    And here is the classes and property list:
    {classes_list}

    return selected properties related to the question in this exact structure with no other comments. Always selected at least one property in each class

    class1[property1,property2,property3],class2[property1,property2]

    question:
    {question}
    """,
    temperature
    )
    return call.lstrip()

#converts output of get_selection into the ? in the example sparql query
def convert_section_to_question(selection):
    """
    This function coverts the selected properties to ?property format for the example sparql query
    Args:
        selection (string): selected properties
    Returns:
        str: a formated string of properties in the question format
    """
    stack=[]
    word=""
    ans=""
    for x in selection:
        if len(stack)!=0:
            if x==',':
                ans=ans+"?"+word.lstrip().split(":",1)[1]+" "
                word=""
            else:
                word=word+x
        if x==',':
            continue
        elif x=='[':
            stack.append('a')
        elif x==']':
            stack.pop()
            ans=ans+"?"+word.lstrip().split(":",1)[1][:-1]+" "
            word=""
    return ans

#converts output of get_secection into the optional block in the example query
def convert_section_to_optional(selection):
    """
    This function coverts the selected properties to optional block format
    Args:
        selection (string): selected properties
    Returns:
        str: a formated string of properties in the optional block format
    """
    stack=[]
    ans=""
    current_class=""
    word=""
    for x in selection:
        if x==']':
            stack.pop()
            ans=ans+f"\tOPTIONAL {{ ?{current_class.lower()} {word} ?{word.lstrip().split(':',1)[1]} . }}\n"
            word=""
        elif len(stack)!=0:
            if x==',':
                ans=ans+f"\tOPTIONAL {{ ?{current_class.lower()} {word} ?{word.lstrip().split(':',1)[1]} . }}\n"
                word=""
            else:
                word=word+x
        elif x==',':
            current_class=""
        elif x=='[':
            stack.append('a')
        else:
            current_class=current_class+x
    return ans

#creates example query to feed through final LLM to help create final query
def create_example_query(path,questions,optional):
    """
    This function takes path, questions and optional blocks and creates the dynamic example query
    Args:
        path (string): class path
        questions (string): selected properties with ?
        optional (string): optional block of all selected properties
    Returns:
        str: the example query to give the last run through of the LLM
    """
    return f"""PREFIX flat: <http://example.org/flat#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX se: <http://example.org/myontology#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT {questions}
WHERE {{
{path}
{optional}
}}"""

#creates sparql query
def create_sparql_query(example,question,temperature):
    """
    This function runs the example query and question through the final LLM with query construction rules
    Args:
        example (string): example query made dynamicly 
        question (string): user's question
    Returns:
        str: the final query to run through GraphDB
    """
    system="you are a helpful assistant that creates a sparql query given a question. You are given an example SPARQL Query telling you what classes and properties to use. You are also given query construction rules that you must follow."
    prompt=f"""Convert natural language questions to SPARQL queries for an Ontotext GraphDB knowledge graph.  
The output must include a correctly formatted SPARQL query using the standard prefixes.  
Do not output any explanatory text outside of the SPARQL query block.

###Example SPARQL Query
{example}

## QUERY CONSTRUCTION RULES

- Reference different classes using `?condition flat:patient_reference se:`
- Always include all properties defined in the template  
- Always lowercase values before applying filters  

- **When filtering ICD codes:**
  - Always filter on `skos:notation`, not the ICD resource variable  
  - Always include both triples together:
    ```
    ?condition (flat:icd9_reference|flat:icd10cm_reference) ?code .
    ?code skos:notation ?notation .
    FILTER(LCASE(?notation) IN (...))
    ```
  - Never filter directly on `?code`

- If a question involves either an ICD-9 or ICD-10 code, use a predicate path:  
  `?condition (flat:icd9_reference|flat:icd10cm_reference) ?code .`

- **When filtering by “WITHOUT” or “NOT having” a condition:**  
  - Use `FILTER NOT EXISTS` with a separate encounter variable  
  - The `NOT EXISTS` block must check all possible encounters for that patient  
  - Example pattern:  
    ```
    FILTER NOT EXISTS {{
        ?encounterX flat:patient_reference ?patient .
        ?condition flat:encounter_reference ?encounterX .
        [condition criteria here]
    }}
    ```

- **Recognize AND vs OR logic in questions:**  
  - “diagnosis A AND diagnosis B” → patient must have both → use separate condition blocks  
  - “diagnosis A OR diagnosis B” → patient must have at least one → use one block with `FILTER IN`  
  - “any diagnosis [category1; category2] AND diagnosis [category3]” → requires (cat1 OR cat2) **and** cat3 → use two blocks  
  - Use FILTER EXISTS when necessary

- **Optimize query efficiency:**  
  - Avoid unnecessary OPTIONALs and repeated triple patterns  
  - Minimize correlated subqueries where possible  
  - Consider pre-filtering or using intermediate variables to reduce join sizes  
  - Avoid functions (like `LCASE`) on large columns if values are already normalized
  


### QUESTION
{question}
"""
    return remove_first_and_last_line(gpt_call(system,prompt,temperature))

#extracts a single collumn from the GraphDB csv file output for metrics
def column_to_list(csv_file):
    reader = csv.reader(csv_file)
    header = next(reader)   
    values = []
    for row in reader:         
        values.append(row[0]) 
    return values

