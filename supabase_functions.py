import streamlit as st
from supabase import create_client, Client
from typing import Optional, Dict, List
from datetime import datetime
import uuid

# For vectorstore functionality:
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

@st.cache_resource
def init_connection() -> Client:
    """
    Initialize and return a Supabase client.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

# Initialize a module-level connection (cached by Streamlit)
supabase = init_connection()

@st.cache_resource
def get_vectorstore():
    """
    Load documents from a file and create a Supabase-based vectorstore.
    """
    loader = TextLoader("utils/clinic_sop.txt")  # Adjust path as needed
    docs = loader.load()
    texts = []
    for doc in docs:
        lines = doc.page_content.split("\n")
        question_answer = []
        for line in lines:
            if line.startswith("Q:"):
                if question_answer:
                    texts.append(Document(page_content="".join(question_answer)))
                question_answer = [line]
            elif line.startswith("A:"):
                question_answer.append(line)
        if question_answer:
            texts.append(Document(page_content="".join(question_answer)))
    embeddings = OpenAIEmbeddings(api_key=st.session_state["OPENAI_API_KEY"])
    vectorstore = SupabaseVectorStore.from_documents(
        texts,
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )
    return vectorstore

def get_patient_by_nric(nric: str) -> Optional[Dict]:
    """
    Retrieve a patient record using the provided NRIC.
    """
    try:
        response = supabase.table("patient").select("*").eq("nric", nric).execute()
        data = response.data
        if data and len(data) > 0:
            return data[0]
        return None
    except Exception as e:
        st.error(f"Error retrieving patient: {e}")
        return None

def get_patient_charges(patient_id: str) -> List[Dict]:
    """
    Retrieve charges for a given patient.
    """
    try:
        response = supabase.table("charge").select("*").eq("patient_id", patient_id).execute()
        return response.data
    except Exception as e:
        st.error(f"Error retrieving charges: {e}")
        return []

def create_payment_record(patient_id: str, amount: float, payment_id: Optional[str] = None) -> Optional[str]:
    """
    Create a payment record for the patient.
    """
    payment_id = payment_id or str(uuid.uuid4())
    now = datetime.now().isoformat()
    data = {
        "payment_id": payment_id,
        "patient_id": patient_id,
        "amount": amount,
        "status": "pending",
        "created_at": now,
        "updated_at": now
    }
    try:
        supabase.table("payment").insert(data).execute()
        return payment_id
    except Exception as e:
        print(f"Error creating payment record: {e}")
        return None

def update_payment_status(payment_id: str, status: str = "completed") -> bool:
    """
    Update the payment status for a given payment record.
    """
    now = datetime.now().isoformat()
    try:
        response = supabase.table("payment").update({
            "status": status,
            "updated_at": now
        }).eq("payment_id", payment_id).execute()
        return bool(response.data)
    except Exception as e:
        st.error(f"Error updating payment status: {e}")
        return False

def update_charges_status(patient_id: str, status: str = "paid") -> bool:
    """
    Update the charge status for a patient.
    """
    now = datetime.now().isoformat()
    try:
        response = supabase.table("charge").update({
            "status": status,
            "updated_at": now
        }).eq("patient_id", patient_id).execute()
        return bool(response.data)
    except Exception as e:
        st.error(f"Error updating charges status: {e}")
        return False

def get_appointment_by_patient_id(patient_id: str) -> Optional[Dict]:
    """
    Retrieve the most recent upcoming appointment for a patient.
    """
    try:
        current_date = datetime.now().date().isoformat()
        response = supabase.table("appointment") \
            .select("*") \
            .eq("patient_id", patient_id) \
            .gte("appointment_date", current_date) \
            .execute()
        data = response.data
        if data and len(data) > 0:
            sorted_appointments = sorted(
                data, 
                key=lambda a: (a["appointment_date"], a.get("appointment_time", "00:00:00")),
                reverse=True
            )
            return sorted_appointments[0]
        return None
    except Exception as e:
        st.error(f"Error retrieving appointment: {e}")
        return None

def update_appointment_datetime(appointment_id: str, new_date: str, new_time: str) -> bool:
    """
    Update the date and time of an appointment.
    """
    try:
        response = supabase.table("appointment").update({
            "appointment_date": new_date,
            "appointment_time": new_time,
            "updated_at": datetime.now().isoformat()
        }).eq("appointment_id", appointment_id).execute()
        return response.data is not None and len(response.data) > 0
    except Exception as e:
        st.error(f"Error updating appointment: {e}")
        return False
