import os
import uuid
from pydantic import BaseModel, Field
from openai import OpenAI
from langsmith import wrappers, Client
from typing import Dict, List
import os
import streamlit as st
from supabase import create_client, Client
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import TextLoader

# Initialize clients
client = Client()
openai_client = wrappers.wrap_openai(OpenAI())

def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

# Use the real connection when initializing the vectorstore
def get_vectorstore():
    loader = TextLoader("utils/clinic_sop.txt")  # Your file path
    docs = loader.load()

    # Split the documents into chunks based on question-answer pairs
    texts = []
    for doc in docs:
        lines = doc.page_content.split("\n")
        question_answer = []
        for line in lines:
            if line.startswith("Q:"):  # This identifies a question
                if question_answer:
                    texts.append(Document(page_content="".join(question_answer)))
                question_answer = [line]  # Start a new question-answer pair
            elif line.startswith("A:"):  # This identifies an answer
                question_answer.append(line)  # Add the answer to the current question
        if question_answer:
            texts.append(Document(page_content="".join(question_answer)))

    # Initialize the real Supabase client
    supabase = init_connection()

    # Create a Supabase vector store with the real Supabase client
    embeddings = OpenAIEmbeddings()
    vectorstore = SupabaseVectorStore.from_documents(
        texts,
        embedding=embeddings,
        client=supabase,  # Use the real Supabase client
        table_name="documents",
        query_name="match_documents"
    )

    return vectorstore


# Perform the FAQ query using the vector store
def faq_query(query):
    vectorstore = get_vectorstore()  # Get the vector store
    try:
        # Run the similarity search on the query
        docs = vectorstore.similarity_search(query, k=1)  # Top 2 matching results
        answer = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        answer = f"Error fetching FAQ: {str(e)}"
    return answer


# Enhanced RAG function that returns both answer and documents
def rag_bot(question: str) -> Dict:
    """
    Enhanced RAG bot that returns both the answer and documents.
    """
    try:
        # Get vector store
        vectorstore = get_vectorstore()
        
        # Perform similarity search
        docs = vectorstore.similarity_search(question, k=1)
        
        # Get the answer using your existing function
        answer = faq_query(question)
        
        # Get document text for evaluation
        doc_texts = [doc.page_content for doc in docs]
        
        return {
            "response": answer,
            "documents": doc_texts
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "documents": []
        }

# 1. Use the existing dataset
def get_existing_dataset():
    """Get the existing dataset by ID"""
    dataset_id = uuid.UUID("136bd2b0-e52a-4c08-a239-c53a938a50d8")
    
    try:
        dataset = client.read_dataset(dataset_id=dataset_id)
        print(f"Using existing dataset: {dataset.name} (ID: {dataset.id})")
        return dataset
    except Exception as e:
        print(f"Error accessing dataset: {e}")
        return None

# 2. Define what we're evaluating
def target(inputs: dict) -> dict:
    """
    This is the function that will be evaluated with examples from the dataset.
    """
    # Call our enhanced RAG bot
    return rag_bot(inputs["question"])

# 3. Define evaluators

# Correctness Evaluator
class CorrectnessGrade(BaseModel):
    score: bool = Field(description="Boolean that indicates whether the response is accurate relative to the reference answer")
    explanation: str = Field(description="Explanation of the reasoning behind the score")

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> Dict:
    """
    Evaluator for factual correctness relative to reference answer.
    """
    instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Your output should be a boolean "score" (True if correct, False if not) and an "explanation" of your reasoning."""
    
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""QUESTION: {inputs["question"]}
GROUND TRUTH ANSWER: {reference_outputs["answer"]}
STUDENT ANSWER: {outputs["response"]}"""}
        ],
        response_format=CorrectnessGrade
    )
    
    parsed = response.choices[0].message.parsed
    return {
        "score": 1.0 if parsed.score else 0.0,
        "explanation": parsed.explanation
    }

# Relevance Evaluator
class RelevanceGrade(BaseModel):
    score: bool = Field(description="Boolean that indicates whether the response is relevant to the question")
    explanation: str = Field(description="Explanation of the reasoning behind the score")

def relevance(inputs: dict, outputs: dict) -> Dict:
    """
    Evaluator for answer relevance to the question.
    """
    instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Your output should be a boolean "score" (True if relevant, False if not) and an "explanation" of your reasoning."""
    
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""QUESTION: {inputs["question"]}
STUDENT ANSWER: {outputs["response"]}"""}
        ],
        response_format=RelevanceGrade
    )
    
    parsed = response.choices[0].message.parsed
    return {
        "score": 1.0 if parsed.score else 0.0,
        "explanation": parsed.explanation
    }

# Groundedness Evaluator
class GroundednessGrade(BaseModel):
    score: bool = Field(description="Boolean that indicates whether the response is grounded in the provided documents")
    explanation: str = Field(description="Explanation of the reasoning behind the score")

def groundedness(inputs: dict, outputs: dict) -> Dict:
    """
    Evaluator for whether the answer is grounded in the retrieved documents.
    """
    instructions = """You are a teacher grading a quiz. 
You will be given FACTS and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Your output should be a boolean "score" (True if grounded, False if not) and an "explanation" of your reasoning."""
    
    # Join the documents with newlines
    doc_string = "\n\n".join(outputs["documents"]) if outputs.get("documents") else "No documents were retrieved."
    
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""FACTS: {doc_string}
STUDENT ANSWER: {outputs["response"]}"""}
        ],
        response_format=GroundednessGrade
    )
    
    parsed = response.choices[0].message.parsed
    return {
        "score": 1.0 if parsed.score else 0.0,
        "explanation": parsed.explanation
    }

# Retrieval Relevance Evaluator
class RetrievalRelevanceGrade(BaseModel):
    score: bool = Field(description="Boolean that indicates whether the retrieved documents are relevant to the question")
    explanation: str = Field(description="Explanation of the reasoning behind the score")

def retrieval_relevance(inputs: dict, outputs: dict) -> Dict:
    """
    Evaluator for whether the retrieved documents are relevant to the question.
    """
    instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a set of FACTS provided by the student. 
Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Your output should be a boolean "score" (True if relevant, False if not) and an "explanation" of your reasoning."""
    
    # Join the documents with newlines
    doc_string = "\n\n".join(outputs["documents"]) if outputs.get("documents") else "No documents were retrieved."
    
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""QUESTION: {inputs["question"]}
FACTS: {doc_string}"""}
        ],
        response_format=RetrievalRelevanceGrade
    )
    
    parsed = response.choices[0].message.parsed
    return {
        "score": 1.0 if parsed.score else 0.0,
        "explanation": parsed.explanation
    }

# 4. Run evaluation
def run_evaluation():
    # Get existing dataset
    dataset = get_existing_dataset()
    if not dataset:
        print("Failed to access the dataset. Exiting.")
        return None
    
    # Run the evaluation
    try:
        experiment_results = client.evaluate(
            target,
            data=dataset.name,  # Use the dataset name instead of ID
            evaluators=[
                correctness,
                relevance,
                groundedness,
                retrieval_relevance
            ],
            experiment_prefix="clinic-faq-rag-evaluation",
            max_concurrency=2,  # Reduced concurrency to avoid API rate limits
            metadata={"version": "v1", "model": "RAG system with vector store"}
        )
        
        print(f"Evaluation completed.")
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return None

if __name__ == "__main__":
    # Ensure required API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
        
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("Error: LANGCHAIN_API_KEY environment variable not set")
        exit(1)
    
    # Run the evaluation with the existing dataset
    results = run_evaluation()
    
    # Optionally, get results as a dataframe if available
    if results:
        try:
            import pandas as pd
            df = results.to_pandas()
            print("\nEvaluation Summary:")
            print(f"Total examples: {len(df)}")
            for evaluator in ['correctness', 'relevance', 'groundedness', 'retrieval_relevance']:
                success_rate = df[df['evaluator'] == evaluator]['success'].mean() * 100
                print(f"{evaluator}: {success_rate:.2f}% success")
        except Exception as e:
            print(f"Error generating summary: {e}")