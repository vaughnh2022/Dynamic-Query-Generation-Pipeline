# Dynamic SPARQL Query Generation Pipeline

## Overview
This pipeline translates natural language questions into **SPARQL queries** over a **FHIR-based knowledge graph** linked to **ICD-9** and **ICD-10 ontologies**. It uses **multiple LLM calls** and **dynamic templates** to construct queries for both the knowledge graph and the ontologies.

## Features
- **Dynamic Class & Property Selection** via LLM  
- **Template-Based SPARQL Construction** with optional blocks  
- **Multiple LLM Passes** for class selection, property selection, and final query generation  
- **Evaluation Metrics**: precision, recall, F1 for query output and selected classes  
- **Modular Design**: separate modules for pipeline, validation, and utilities 
