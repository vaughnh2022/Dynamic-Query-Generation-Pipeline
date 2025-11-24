from pipeline import grag_pipeline
from pipeline_validator import pipeline_metrics

#----------------------------------
#  This is the main function that gives you a high level ability to run my pipeline
#
# grag_pipeline allows you to query the pipeline given one question as a string parameter
#     ex: grag_pipeline("how many patients are in the database")
#
# pipeline_metrics allows you to run multiple questions and calculate metrics
#     please note you must follow the structure of the given example-test.csv
#     ex pipeline_metrics("example-test.csv","example-test-output.csv") 
