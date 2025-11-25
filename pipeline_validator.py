
import math
import ast
import pandas as pd
from pipeline import grag_pipeline

#---------------------------
# Metric Validation Functions
# 

#given the structured input from a csv file extracted gold standard classes
def find_extracted_classes(full_list):
    if full_list is None or full_list == "" or (isinstance(full_list, float) and math.isnan(full_list)):
        return []
    ans=[]
    stack=[]
    word=""
    for x in full_list:
        if x=='[':
            stack.append(x)
        if len(stack)==0:
            if x==',':
                ans.append(word)
                word=""
            else:
                word=word+x
        if x==']':
            stack.pop()
    ans.append(word)
    return ans

#clean ground truth for comparisons
def parse_ground_truth(x):
    if pd.isna(x):
        return []

    # If it's already a list, make sure all elements are strings
    if isinstance(x, list):
        return [str(i) for i in x]

    x = str(x).strip()

    # Convert booleans
    if x.lower() == 'true':
        return ["True"]
    if x.lower() == 'false':
        return ["False"]

    # Recursively try to parse stringified list
    while (x.startswith('[') and x.endswith(']')) or (x.startswith('"[') and x.endswith(']"')):
        try:
            # Remove outer quotes if present
            if x.startswith('"[') and x.endswith(']"'):
                x = x[1:-1]  # remove wrapping quotes

            parsed = ast.literal_eval(x)
            # If the result is still a string, loop to try again
            if isinstance(parsed, str):
                x = parsed
            else:
                # Make sure all elements are strings
                return [str(i) for i in parsed]
        except:
            break

    # Fallback: treat as a single-element list
    return [x]

#takes structured csv input and runs pipeline over each question and calculates metrics
def pipeline_metrics(path,output_path,temperature):
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop(df.index[0]).reset_index(drop=True)

    #clean ground truth for comparisons
    df['ground_truth_query_output'] = df['ground_truth_query_output'].apply(parse_ground_truth)

    #pipeline to add a tuple of things into collumns in a pandas dataframe
    print("\n-----Running pipeline for metrics------\n")
    df[["selected_classes", "selected_properties", "pipeline_created_query", "pipeline_query_output"]] = df["question"].apply(
        lambda x: pd.Series(grag_pipeline(x,temperature))
    )
    
    #finds query output exacy match
    df['query_output_exact_match'] = df.apply(
        lambda row: sorted(row['ground_truth_query_output']) == sorted(row['pipeline_query_output']),
        axis=1
    )
    #finds class exact match
    df['extracted_classes'] = df['gold_classes_and_properties'].apply(find_extracted_classes)
    df['class_exact_match'] = df.apply(
        lambda row: sorted(row['extracted_classes']) == sorted(row['selected_classes']),
        axis=1
    )
    #calculates precision and recall for output and classes
    df['answers_and_gold_intersection'] = df.apply(
       lambda row: sorted(set(row['ground_truth_query_output']) & set(row['pipeline_query_output'])),
       axis=1
    )
    df['precision_for_output'] = df.apply(
        lambda row: len(row['answers_and_gold_intersection']) / len(row['ground_truth_query_output']) 
        if len(row['ground_truth_query_output']) > 0 else 0,
        axis=1
    )
    df['recall_for_output'] = df.apply(
        lambda row: len(row['answers_and_gold_intersection']) / len(row['pipeline_query_output']) 
        if len(row['pipeline_query_output']) > 0 else 0,
        axis=1
    )
    f1=0
    for precision, recall in zip(df['precision_for_output'], df['recall_for_output']):
        f1+=(2*precision*recall)/(precision+recall) if (precision+recall)!=0 else 0
    f1=f1/len(df['recall_for_output'])
    df['output_F1']=f1
    df['answers_and_gold_intersection_classes'] = df.apply(
       lambda row: sorted(set(row['extracted_classes']) & set(row['selected_classes'])),
       axis=1
    )
    df['precision_for_classes'] = df.apply(
        lambda row: len(row['answers_and_gold_intersection_classes']) / len(row['extracted_classes']) 
        if len(row['extracted_classes']) > 0 else 0,
        axis=1
    )
    df['recall_for_classes'] = df.apply(
        lambda row: len(row['answers_and_gold_intersection_classes']) / len(row['selected_classes']) 
        if len(row['selected_classes']) > 0 else 0,
        axis=1
    )
    f1=0
    for precision, recall in zip(df['precision_for_classes'], df['recall_for_classes']):
        f1+=(2*precision*recall)/(precision+recall) if (precision+recall)!=0 else 0
    f1=f1/len(df['recall_for_classes'])
    df['classes_F1']=f1
    #finds percentages
    output_percent_true=df['query_output_exact_match'].mean()*100
    class_percent_true=df['class_exact_match'].mean()*100
    df['query_output_exact_match_percentage'] = output_percent_true
    df['class_exact_match_percentage']=class_percent_true
    #drops non-relevant tables
    df.drop(columns=['answers_and_gold_intersection'],inplace=True)
    df.drop(columns=['answers_and_gold_intersection_classes'],inplace=True)
    df.drop(columns=['extracted_classes'],inplace=True)
    #outputs to another csv
    df.to_csv(output_path, index=False, encoding="utf-8")
