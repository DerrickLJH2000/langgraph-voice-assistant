import os
import re
import uuid
import json
import stripe
import qrcode
from PIL import Image
import time
import io
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from typing import Dict, Optional, List, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from supabase import create_client, Client
import streamlit.components.v1 as components
from streamlit import html

# OpenAI and ElevenLabs imports for ASR and TTS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from pydantic import BaseModel, Field

# ElevenLabs & OpenAI client initialization for audio processing
from elevenlabs import play, VoiceSettings
from elevenlabs.client import ElevenLabs
import openai

# Load Hugging Face ASR model and processor
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from supabase_functions import (
    get_vectorstore,
    get_patient_by_nric,
    get_patient_charges,
    create_payment_record,
    update_payment_status,
    update_charges_status,
    get_appointment_by_patient_id,
    update_appointment_datetime
)
from appointment_agent import *

processor = AutoProcessor.from_pretrained("jensenlwt/whisper-small-singlish-122k")
model = AutoModelForSpeechSeq2Seq.from_pretrained("jensenlwt/whisper-small-singlish-122k")
# Initialize clients for ASR and TTS
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment
openai_client = openai  # Using openai python library for Whisper transcription
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
stripe.api_key = os.getenv("STRIPE_API_KEY")

# ----------------------- Automatic Speech Recognition (ASR) -----------------------
def record_audio_input(silence_threshold: float = 500, silence_duration: float = 1.0, sample_rate: int = 16000) -> str:
    """
    Enhanced version of the audio recording function with visual feedback.
    
    Parameters:
      silence_threshold: RMS threshold below which the audio is considered silent.
      silence_duration: Amount of silence (in seconds) needed to stop recording.
      sample_rate: Sampling rate for recording.
    
    Returns:
      The transcribed text from the audio using OpenAI's Whisper model.
    """
    import sounddevice as sd
    import numpy as np
    import io
    from scipy.io.wavfile import write
    import time

    # Define chunk parameters
    chunk_duration = 0.5  # seconds per chunk
    chunk_samples = int(chunk_duration * sample_rate)
    silence_chunks_needed = int(silence_duration / chunk_duration)

    audio_chunks = []
    consecutive_silence = 0
    
    # Create a placeholder for the recording status
    status_placeholder = st.empty()
    status_placeholder.info("🎤 Recording... Speak now; recording will stop after a period of silence.")
    
    # Create a placeholder for the audio level visualization
    level_placeholder = st.empty()
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        recording_active = True
        start_time = time.time()
        
        while recording_active:
            chunk, _ = stream.read(chunk_samples)
            audio_chunks.append(chunk)
            
            # Calculate RMS (energy) for the chunk
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            
            # Update the audio level visualization
            level = min(int(rms / 100), 20)  # Limit to 20 bars
            level_bar = "🟩" * level + "⬜" * (20 - level)
            level_placeholder.markdown(f"Audio level: {level_bar}")
            
            if rms < silence_threshold:
                consecutive_silence += 1
                status_placeholder.info(f"🎤 Detecting silence... ({consecutive_silence}/{silence_chunks_needed})")
            else:
                consecutive_silence = 0
                status_placeholder.info("🎤 Recording... Speak clearly. Will stop after silence is detected.")
            
            # If enough consecutive silent chunks are detected, stop recording.
            if consecutive_silence >= silence_chunks_needed:
                recording_active = False
            
            # Also provide a maximum recording time (30 seconds)
            if time.time() - start_time > 30:
                status_placeholder.warning("Maximum recording time reached (30 seconds).")
                recording_active = False

    # Update status
    status_placeholder.info("⏳ Processing audio... Transcribing your speech.")
    level_placeholder.empty()
    
    # Combine all chunks into one array.
    audio_data = np.concatenate(audio_chunks, axis=0)
    
    # Save the audio data to an in-memory WAV file.
    audio_bytes = io.BytesIO()
    write(audio_bytes, sample_rate, audio_data)
    audio_bytes.seek(0)
    audio_bytes.name = "audio.wav"

    # Transcribe using OpenAI Whisper (model "whisper-1")
    transcription = openai_client.audio.transcriptions.create(
       model="whisper-1", 
       language="en",
       file=audio_bytes,
    )
    
    # Clear the status placeholders
    status_placeholder.empty()
    
    transcription_text = transcription.text.strip()
    if transcription_text:
        status_placeholder.success(f"Transcription: {transcription_text}")
    else:
        status_placeholder.error("No speech detected. Please try again.")
    
    return transcription_text

# ----------------------- Text-to-Speech (TTS) -----------------------
def play_audio_response(text: str):
    """
    Converts text to speech using ElevenLabs and plays the audio using Streamlit's audio player.
    """
    voice_settings = VoiceSettings(
        stability=0.0,
        similarity_boost=1.0,
        style=0.0,
        use_speaker_boost=True,
    )
    # Convert text to speech (this returns a generator)
    tts_generator = elevenlabs_client.text_to_speech.convert(
        voice_id="6qpxBH5KUSDb40bij36w",  # Replace with your desired voice ID
        output_format="mp3_44100_128",
        text=text,
        model_id="eleven_multilingual_v2", 
        voice_settings=voice_settings,
    )
    # Collect the generator's output into a single bytes object.
    tts_audio = b"".join(tts_generator)
    st.audio(tts_audio, format="audio/mp3")


# ----------------------- Intent Classification -----------------------
class IntentClassification(BaseModel):
    """Classification of user intent."""
    intent: str = Field(
        description="The classified intent, must be one of: 'appointment', 'payment', 'faq', 'unknown'")
    confidence: float = Field(
        description="Confidence score between 0 and 1")
    explanation: str = Field(
        description="Brief explanation of why this intent was chosen")


vectorstore = get_vectorstore()

# ----------------------- Stripe API Functions -----------------------
def load_products(json_filepath: str) -> list:
    """
    Load products from JSON file with improved error handling.
    
    Parameters:
        json_filepath: Path to the JSON file containing product data
        
    Returns:
        list: List of product dictionaries
    """
    import json
    import os
    
    try:
        # Check if file exists
        if not os.path.exists(json_filepath):
            print(f"Warning: Products file not found at {json_filepath}")
            return []
            
        with open(json_filepath, "r") as f:
            products = json.load(f)
            
        # Validate products structure
        if not isinstance(products, list):
            print(f"Warning: Products file does not contain a list")
            return []
            
        # Validate each product has required fields
        valid_products = []
        for i, product in enumerate(products):
            if not isinstance(product, dict):
                print(f"Warning: Product at index {i} is not a dictionary")
                continue
                
            if "id" not in product:
                print(f"Warning: Product at index {i} missing 'id' field")
                continue
                
            if "name" not in product:
                print(f"Warning: Product with ID {product.get('id')} missing 'name' field")
                continue
                
            valid_products.append(product)
            
        print(f"Loaded {len(valid_products)} valid products from {json_filepath}")
        
        # Debug output to help identify issues
        product_ids = [p.get("id") for p in valid_products]
        print(f"Available product IDs: {product_ids}")
        
        return valid_products
        
    except json.JSONDecodeError as e:
        print(f"Error parsing products JSON file: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error loading products: {e}")
        return []

def match_charge_to_product(charge_description: str, products: list) -> Optional[str]:
    charge_description_lower = charge_description.lower()
    for product in products:
        if product["name"].lower() in charge_description_lower:
            return product["id"]
    return None

def create_line_items_from_charges(charges: list, products: list) -> list:
    line_items = []
    for charge in charges:
        description = charge.get("description", "")
        product_id = match_charge_to_product(description, products)
        if not product_id:
            print(f"Warning: No matching product found for charge '{description}'. Skipping.")
            continue
        unit_price = float(charge.get("unit_price", 0))
        quantity = int(charge.get("quantity", 1))
        price = stripe.Price.create(
            currency="sgd",
            unit_amount=int(unit_price * 100),
            product=product_id,
        )
        line_items.append({"price": price.id, "quantity": quantity})
    return line_items

def create_checkout_session(charges: list) -> dict:
    """
    Create a Stripe checkout session for the given charges.
    
    Parameters:
        charges: A list of charge dictionaries
        
    Returns:
        dict: The Stripe checkout session object
    """
    if not charges or len(charges) == 0:
        raise ValueError("No charges provided")
        
    try:
        products = load_products("utils/products.json")
        line_items = create_line_items_from_charges(charges, products)
        
        if not line_items:
            raise ValueError("No valid line items could be created from charges.")
            
        session = stripe.checkout.Session.create(
            payment_method_types=["card", "paynow"],
            line_items=line_items,
            mode="payment",
            success_url="https://example.com/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://example.com/cancel",
        )
        
        # Verify the session has the required attributes
        if not hasattr(session, 'id') or not hasattr(session, 'url'):
            raise ValueError("Created session is missing required attributes")
            
        return session
    except Exception as e:
        st.error(f"Error creating checkout session: {e}")
        # Return None instead of raising to allow for graceful error handling
        return None

# 1. Modify the check_payment_status function to handle None values
def check_payment_status(checkout_session_id: str) -> str:
    """
    Check the payment status of a Stripe checkout session.
    
    Parameters:
        checkout_session_id: The ID of the Stripe checkout session
        
    Returns:
        str: 'completed' if the payment was successful, 'pending' otherwise
    """
    if not checkout_session_id:
        return "pending"  # Return pending if no session ID is provided
        
    try:
        session = stripe.checkout.Session.retrieve(checkout_session_id)
        return "completed" if session.payment_status == "paid" else "pending"
    except Exception as e:
        st.error(f"Error checking payment status: {e}")
        return "pending"  # Default to pending if there's an error

def save_qr_code_image(qr_image: Image, directory: str = "static/qr_codes") -> str:
    import os
    from datetime import datetime
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qr_code_{timestamp}.png"
    file_path = os.path.join(directory, filename)
    qr_image.save(file_path)
    return file_path

def create_payment_qr_code(payment_link_url: str) -> tuple[Image.Image, str]:
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(payment_link_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color=(128, 153, 165))
    logo = Image.open("utils/vocare_logo.PNG")
    logo = logo.resize((50, 50))
    img_w, img_h = img.size
    logo_w, logo_h = logo.size
    pos = ((img_w - logo_w) // 2, (img_h - logo_h) // 2)
    img.paste(logo, pos)
    qr_path = save_qr_code_image(img)
    return img, qr_path

# ----------------------- Simple NRIC Verification -----------------------
def verify_nric_simple(user_input: str) -> Dict:
    nric_pattern = r'[sStT]\d{7}[a-zA-Z]'
    match = re.search(nric_pattern, user_input)
    if not match:
        return {"verified": False, "message": "No valid NRIC found. Please provide your NRIC (e.g., S1234567A)."}
    nric = match.group(0).upper()
    patient = get_patient_by_nric(nric)
    if patient:
        welcome_msg = f"Good Day {patient['first_name']}! Would you like to schedule an appointment or make payment?"
        return {"verified": True, "patient": patient, "nric": nric, "message": welcome_msg}
    else:
        return {"verified": False, "nric": nric, "message": "NRIC not found. Please check and try again."}

# ----------------------- Define Workflow State -----------------------
class ClinicState(TypedDict):
    messages: Annotated[List, None]
    verified: bool
    patient: Optional[Dict]
    nric: Optional[str]
    service: Optional[str]
    stage: str
    last_processed_msg_idx: int
    appointment_stage: Optional[str]
    current_appointment: Optional[Dict]
    offered_slots: Optional[List[Dict]]
    payment_stage: Optional[str]
    payment_id: Optional[str]
    payment_amount: Optional[float]
    payment_completed: Optional[bool]
    chat_started: Optional[bool]
    stripe_checkout_session_id: Optional[str]
    stripe_checkout_session_url: Optional[str]
    qr_code_path: Optional[str]
    intent_classification: Optional[Dict]

# ----------------------- NRIC Verifier Node -----------------------
def nric_verifier_node(state: ClinicState) -> ClinicState:
    if not state.get("verified"):
        current_msg_count = len(state["messages"])
        last_processed = state.get("last_processed_msg_idx", -1)
        if current_msg_count > last_processed + 1:
            for i in range(last_processed + 1, current_msg_count):
                if state["messages"][i]["role"] == "user":
                    user_input = state["messages"][i]["content"]
                    result = verify_nric_simple(user_input)
                    state["verified"] = result.get("verified", False)
                    state["patient"] = result.get("patient")
                    state["nric"] = result.get("nric")
                    state["messages"].append({"role": "assistant", "content": result.get("message", "")})
                    state["last_processed_msg_idx"] = current_msg_count
                    break
    if state["verified"]:
        state["stage"] = "service_selection"
    return state

# ----------------------- Appointment Agent Node -----------------------
# 1. Add the appointment selection processing to your appointment_agent_node
def appointment_agent_node(state: ClinicState) -> ClinicState:
    # First, check if we're in the selection stage and need to process a slot choice
    if state.get("appointment_stage") == "selecting_slot":
        # Find the last user message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
                
        if last_user_message:
            # Extract number from user message (handles both "1" and "one" formats)
            slot_number = extract_number_from_text(last_user_message)
            
            if 1 <= slot_number <= 3:  # Valid slot selection
                slot_index = slot_number - 1
                if state.get("offered_slots") and slot_index < len(state.get("offered_slots", [])):
                    selected_slot = state["offered_slots"][slot_index]
                    if "current_appointment" in state:
                        appointment_id = state["current_appointment"]["appointment_id"]
                        success = update_appointment_datetime(appointment_id, selected_slot["date"], selected_slot["time"])
                        if success:
                            state["messages"].append({"role": "assistant", "content": f"Your appointment has been rescheduled to {selected_slot['date']} at {selected_slot['time']}."})
                        else:
                            state["messages"].append({"role": "assistant", "content": "I wasn't able to update your appointment. Please try again later."})
                    
                    # Clear the appointment stage and related data
                    state["appointment_stage"] = None
                    if "offered_slots" in state: del state["offered_slots"]
                    if "current_appointment" in state: del state["current_appointment"]
                    return state  # Return early since we processed the selection
                else:
                    state["messages"].append({"role": "assistant", "content": "There was an issue with your selection. Please try again."})
                    return state
            elif last_user_message.strip() and not any(msg["content"].startswith("I didn't recognize") for msg in reversed(state["messages"][:5]) if msg["role"] == "assistant"):
                # Only add the error message if we haven't already done so recently
                state["messages"].append({
                    "role": "assistant", 
                    "content": "I didn't recognize that as a slot selection. Please choose one of the provided options by saying 'one', 'two', or 'three', or the corresponding number."
                })
                return state
    
    # If we reach here, either we're not in selection stage, or we didn't find a valid slot selection
    # Continue with the original function logic for presenting appointment options
    
    patient = state.get("patient", {})
    patient_id = patient.get("patient_id") if patient else None
    if not patient_id:
        state["messages"].append({"role": "assistant", "content": "I need your patient information to schedule appointments."})
        return state
        
    appointment = get_appointment_by_patient_id(patient_id)
    if appointment:
        appt_date = appointment.get("appointment_date")
        if appt_date is not None and not isinstance(appt_date, str):
            appt_date = appt_date.strftime('%Y-%m-%d')
        elif appt_date is None:
            appt_date = "Unknown"
            
        try:
            appt_date_obj = datetime.strptime(appt_date, '%Y-%m-%d').date()
        except Exception:
            appt_date_obj = datetime.now().date()
            
        slot1 = (appt_date_obj, "09:30:00")
        slot2 = (appt_date_obj, "14:30:00")
        slot3 = (appt_date_obj + timedelta(days=1), "09:30:00")
        
        options_text = (
            f"You have an appointment scheduled for {appt_date}.\n"
            "Here are the available time slots for the appointment:\n"
            f"1. {slot1[0].strftime('%Y-%m-%d')} at {slot1[1]}\n"
            f"2. {slot2[0].strftime('%Y-%m-%d')} at {slot2[1]}\n"
            f"3. {slot3[0].strftime('%Y-%m-%d')} at {slot3[1]}\n\n"
            "Please choose a slot by saying the option number (such as 'one', 'two', or 'three')."
        )
        state["messages"].append({"role": "assistant", "content": options_text})
        state["offered_slots"] = [
            {"date": slot1[0].strftime('%Y-%m-%d'), "time": slot1[1]},
            {"date": slot2[0].strftime('%Y-%m-%d'), "time": slot2[1]},
            {"date": slot3[0].strftime('%Y-%m-%d'), "time": slot3[1]}
        ]
        state["appointment_stage"] = "selecting_slot"
        state["current_appointment"] = dict(appointment)
    else:
        state["messages"].append({"role": "assistant", "content": "You have no upcoming appointments. Restarting to the main page..."})
        time.sleep(5)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
    return state

# ----------------------- Payment Agent Node -----------------------
# 3. Improve the payment agent node to handle errors better
def payment_agent_node(state: ClinicState) -> ClinicState:
    patient = state.get("patient", {})
    patient_id = patient.get("patient_id") if patient else None
    
    if not patient_id:
        state["messages"].append({"role": "assistant", "content": "I need your patient info to process payment."})
        return state
        
    if state.get("payment_completed"):
        if "Payment received" not in state["messages"][-1]["content"]:
            state["messages"].append({"role": "assistant", "content": "Payment has already been completed. Thank you!"})
        return state
        
    if state.get("payment_stage") != "ready_for_payment":
        # Fetch charges
        charges = get_patient_charges(patient_id)
        if not charges or len(charges) == 0:
            state["messages"].append({"role": "assistant", "content": "You don't have any pending charges."})
            return state
            
        # Calculate total
        total = sum(float(charge.get("unit_price", 0)) * float(charge.get("quantity", 0)) for charge in charges)
        
        # Format charge list
        charge_list = "\n".join([
            f"- {charge.get('description', 'Unknown service')}: "
            f"${float(charge.get('unit_price', 0)):.2f} \u00D7 {int(charge.get('quantity', 0))} = "
            f"${float(charge.get('unit_price', 0)) * float(charge.get('quantity', 0)):.2f}"
            for charge in charges
        ])
        
        # Create payment record
        payment_id = create_payment_record(patient_id, total)
        state["payment_id"] = payment_id
        state["payment_amount"] = total
        
        # Update charge status
        try:
            update_charges_status(patient_id, "pending")
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error updating charges: {e}"})
            
        # Create checkout session
        try:
            # Only create checkout session if we have required parameters
            if payment_id and charges and len(charges) > 0:
                checkout_session = create_checkout_session(charges)
                if checkout_session and checkout_session.id and checkout_session.url:
                    state["stripe_checkout_session_id"] = checkout_session.id
                    state["stripe_checkout_session_url"] = checkout_session.url
                    qr_image, qr_path = create_payment_qr_code(checkout_session.url)
                    state["qr_code_path"] = qr_path
                else:
                    # Handle case when checkout session was created but is missing required attributes
                    state["messages"].append({"role": "assistant", "content": "Unable to create complete payment session. Please try again or pay at the front desk."})
            else:
                # Handle case when we don't have required parameters
                state["messages"].append({"role": "assistant", "content": "Unable to create payment session due to missing information. Please try again or pay at the front desk."})
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error creating checkout session: {e}"})
            
        # Update state
        state["payment_stage"] = "ready_for_payment"
        
        # Prepare payment message
        payment_message = f"Here are your current charges:\n\n{charge_list}\n\nTotal amount due: ${total:.2f}\n\n"
        
        if state.get("stripe_checkout_session_url"):
            payment_message += "You can make a payment using our secure payment portal. I've generated a payment link and QR code for you."
        else:
            payment_message += "You can pay at the front desk. Please mention your NRIC."
            
        state["messages"].append({"role": "assistant", "content": payment_message})
        
    return state

# ----------------------- FAQ Agent Node -----------------------
def faq_agent_node(state: ClinicState) -> ClinicState:
    last_user_message = None
    for i in range(len(state["messages"])-1, -1, -1):
        if state["messages"][i]["role"] == "user":
            last_user_message = state["messages"][i]["content"]
            break
    if not last_user_message:
        if state.get("service") == "faq_agent" and not any("answer your questions" in m["content"] for m in state["messages"] if m["role"]=="assistant"):
            state["messages"].append({"role": "assistant", "content": "I'm here to answer your questions. What would you like to know?"})
        return state
    if "kiosk" in last_user_message.lower() and "new patient" in last_user_message.lower():
        state["messages"].append({"role": "assistant", "content": "New patients should register at the front desk first. Then you can use the kiosk for future visits."})
        return state
    try:
        query = last_user_message
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=1)
        if docs and len(docs) > 0:
            answers = []
            for doc in docs:
                content = doc.page_content
                if "Q:" in content and "A:" in content:
                    answer_part = content.split("A:")[1].strip() if len(content.split("A:")) > 1 else content
                    answers.append(answer_part)
                else:
                    answers.append(content)
            response = "\n\n".join(answers)
            state["messages"].append({"role": "assistant", "content": response})
        else:
            state["messages"].append({"role": "assistant", "content": "I'm sorry, I couldn't find information related to your question."})
    except Exception as e:
        state["messages"].append({"role": "assistant", "content": "I'm having trouble retrieving the information right now. Please try again."})
        print(f"Error in FAQ agent: {e}")
    return state

# ----------------------- Service Router Node -----------------------
def service_router_node(state: ClinicState) -> ClinicState:
    # Skip processing for completed appointment stage
    if state.get("stage") == "completed_appointment":
        return state
        
    if state.get("stage") != "service_selection":
        return state
        
    if state.get("service") is None:
        return state
        
    user_input = state["service"]
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        )
        parser = PydanticOutputParser(pydantic_object=IntentClassification)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI assistant for a medical clinic. 
Classify the user's intent into one of the following categories:
- 'appointment': For booking, scheduling, or changing appointments
- 'payment': For making payments or inquiring about bills
- 'faq': For general questions about the clinic
- 'unknown': Otherwise
Respond with a valid JSON object including intent, confidence, and explanation.
"""),
            HumanMessage(content=f"User input: {user_input}")
        ])
        formatted_prompt = prompt | llm | parser
        result = formatted_prompt.invoke({})
        state["intent_classification"] = {"intent": result.intent, "confidence": result.confidence, "explanation": result.explanation}
        if result.intent == "appointment":
            state["service"] = "appointment_agent"
            state["messages"].append({"role": "assistant", "content": "I'll help you with your appointment."})
        elif result.intent == "payment":
            state["service"] = "payment_agent"
            state["messages"].append({"role": "assistant", "content": "I'll help you with your payment."})
        elif result.intent == "faq":
            state["service"] = "faq_agent"
            state["messages"].append({"role": "assistant", "content": "I'll answer your questions about our clinic."})
        else:
            return state
    except Exception as e:
        st.error(f"Error classifying intent: {e}")
        lower_input = user_input.lower()
        if any(word in lower_input for word in ["appointment", "schedule", "book"]):
            state["service"] = "appointment_agent"
            state["messages"].append({"role": "assistant", "content": "I'll help you with your appointment."})
        elif any(word in lower_input for word in ["payment", "pay", "bill"]):
            state["service"] = "payment_agent"
            state["messages"].append({"role": "assistant", "content": "I'll help you with your payment."})
        elif any(word in lower_input for word in ["question", "info", "help", "what", "how", "where", "when", "who", "why"]) or "?" in lower_input:
            state["service"] = "faq_agent"
            state["messages"].append({"role": "assistant", "content": "I'll answer your questions about our clinic."})
        else:
            state["messages"].append({"role": "assistant", "content": "I'm sorry, I didn't understand."})
            return state
    state["stage"] = "routed"
    return state


# ----------------------- Build Extended Workflow -----------------------
def build_workflow_extended():
    graph_builder = StateGraph(ClinicState)
    graph_builder.add_node("nric_verifier", nric_verifier_node)
    graph_builder.add_node("service_router", service_router_node)
    graph_builder.add_node("appointment_agent", appointment_agent_node)
    graph_builder.add_node("payment_agent", payment_agent_node)
    graph_builder.add_node("faq_agent", faq_agent_node)
    
    def route_after_nric(state: ClinicState) -> str:
        if state.get("verified"):
            return "service_router"
        return "end"
    
    graph_builder.add_conditional_edges("nric_verifier", route_after_nric, {"service_router": "service_router", "end": END})
    
    def route_service(state: ClinicState) -> str:
        # Special case for completed appointment
        if state.get("stage") == "completed_appointment":
            return "end"
            
        service = state.get("service", "")
        if service == "appointment_agent":
            return "appointment_agent"
        elif service == "payment_agent":
            return "payment_agent"
        elif service == "faq_agent":
            return "faq_agent"
        return "end"
    
    graph_builder.add_conditional_edges("service_router", route_service, {
        "appointment_agent": "appointment_agent",
        "payment_agent": "payment_agent",
        "faq_agent": "faq_agent",
        "end": END
    })
    
    graph_builder.add_edge("faq_agent", END)
    graph_builder.add_edge("appointment_agent", END)
    graph_builder.add_edge("payment_agent", END)
    graph_builder.set_entry_point("nric_verifier")
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

# ----------------------- Streamlit UI -----------------------
def add_fixed_chat_input_css():
    st.markdown("""
    <style>
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: white;
        padding: 15px;
        border-top: 1px solid #ddd;
        box-shadow: 0px -5px 10px rgba(0, 0, 0, 0.05);
    }
    .chat-container {
        margin-bottom: 100px;
        padding-bottom: 20px;
    }
    .recording-message {
        color: #666;
        font-style: italic;
        margin-top: 5px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("Clinic Voice Assistant")
    add_fixed_chat_input_css()
    
    # Initialize chat if not already started
    if "chat_started" not in st.session_state:
        st.session_state["chat_started"] = False

    if not st.session_state["chat_started"]:
        st.markdown("""
        ## Welcome to our Clinic Voice Assistant
        
        This assistant can help you with appointments, payments, or general queries.
        Please speak clearly when prompted.
        """)
        if st.button("Begin Chat", key="begin_chat_btn"):
            st.session_state["chat_started"] = True
            st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! Please say your NRIC number to begin."}]
            st.session_state["verified"] = False
            st.session_state["patient"] = None
            st.session_state["nric"] = None
            st.session_state["service"] = None
            st.session_state["stage"] = "nric_verification"
            st.session_state["last_processed_msg_idx"] = -1
            st.session_state["appointment_stage"] = None
            st.session_state["payment_stage"] = None
            st.session_state["payment_id"] = None
            st.session_state["payment_amount"] = None
            st.session_state["payment_completed"] = False
            st.session_state["stripe_checkout_session_id"] = None
            st.session_state["stripe_checkout_session_url"] = None
            st.session_state["voice_mode"] = True  # Enable voice mode by default
            st.rerun()
    else:
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for msg in st.session_state["messages"]:
                if msg.get("content"):
                    st.chat_message(msg["role"]).markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Build and run workflow
        workflow = build_workflow_extended()
        thread_config = {"configurable": {"thread_id": "1"}}
        
        if st.session_state.get("payment_stage") == "ready_for_payment" and not st.session_state.get("payment_completed"):
            st.markdown("---")
            st.subheader("Payment Portal")
            patient = st.session_state.get("patient", {})
            st.write(f"Total amount due: ${st.session_state.get('payment_amount', 0):.2f}")
            
            # Display payment options
            if st.session_state.get("stripe_checkout_session_url"):
                st.markdown(f"[Complete Payment]({st.session_state.get('stripe_checkout_session_url')})")
                if st.session_state.get("qr_code_path"):
                    qr_image = Image.open(st.session_state.get("qr_code_path"))
                    st.image(qr_image, caption="Scan to pay", width=300)
            
            # Cancel payment button
            if st.button("Cancel Payment", key="cancel_payment_btn"):
                st.session_state["payment_stage"] = None
                st.session_state["stripe_checkout_session_id"] = None
                st.session_state["stripe_checkout_session_url"] = None
                st.session_state["qr_code_path"] = None
                st.session_state["payment_id"] = None
                st.session_state["payment_amount"] = None
                st.session_state["payment_completed"] = False
                st.session_state["messages"].append({"role": "assistant", "content": "Payment process cancelled. Please restart the chat if you wish to try again."})
                st.rerun()
            
            # Payment status check
            st.info("Waiting for payment to complete... (This page auto-refreshes every 5 seconds)")
            st_autorefresh(interval=5000, limit=None, key="payment_autorefresh")
            
            # Check if we have a valid session ID before attempting to check status
            checkout_session_id = st.session_state.get("stripe_checkout_session_id")
            if checkout_session_id:
                status = check_payment_status(checkout_session_id)
                if status == "completed":
                    st.write("Payment detected as completed on Stripe. Updating records...")
                    p_updated = update_payment_status(st.session_state["payment_id"])
                    c_updated = update_charges_status(st.session_state["patient"]["patient_id"])
                    if p_updated and c_updated:
                        st.session_state["payment_completed"] = True
                        st.session_state["payment_stage"] = "completed"
                        st.session_state["messages"].append({"role": "assistant", "content": f"Payment of ${st.session_state.get('payment_amount'):.2f} has been completed successfully via Stripe Checkout!"})
                    else:
                        st.session_state["messages"].append({"role": "assistant", "content": "Payment update failed. Please try again."})
                    st.rerun()
            else:
                st.warning("No payment session created yet or session ID is missing.")
        
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col3:
            if st.button("Restart Chat", key="restart_chat_btn"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            # Toggle between voice and text input
            voice_mode = st.session_state.get("voice_mode", True)
            if st.button("🎤 Voice" if not voice_mode else "⌨️ Text", key="toggle_input_mode"):
                st.session_state["voice_mode"] = not voice_mode
                st.rerun()
                
        with col1:
            voice_mode = st.session_state.get("voice_mode", True)
            
            if voice_mode:
                # Voice input mode
                if st.button("Start Speaking", key="start_speaking_btn"):
                    with st.spinner("Listening..."):
                        transcription = record_audio_input(silence_threshold=500, silence_duration=1.0, sample_rate=16000)
                        
                        if transcription.strip():
                            st.session_state.messages.append({"role": "user", "content": transcription})
                            
                            # Process input based on current stage
                            if st.session_state.get("stage") == "service_selection" and st.session_state.get("service") is None:
                                st.session_state["service"] = transcription.strip()
                            
                            # Update state through workflow
                            current_state = {
                                "messages": st.session_state.messages.copy(),
                                "verified": st.session_state.verified,
                                "patient": st.session_state.patient,
                                "nric": st.session_state.nric,
                                "service": st.session_state.service,
                                "stage": st.session_state.stage,
                                "last_processed_msg_idx": st.session_state.get("last_processed_msg_idx", -1),
                                "appointment_stage": st.session_state.get("appointment_stage"),
                                "current_appointment": st.session_state.get("current_appointment"),
                                "offered_slots": st.session_state.get("offered_slots"),
                                "payment_stage": st.session_state.get("payment_stage"),
                                "payment_id": st.session_state.get("payment_id"),
                                "payment_amount": st.session_state.get("payment_amount"),
                                "payment_completed": st.session_state.get("payment_completed", False),
                                "chat_started": st.session_state.get("chat_started", False)
                            }
                            result_state = workflow.invoke(current_state, config=thread_config)
                            st.session_state.update(result_state)
                            st.rerun()
            else:
                # Text input mode (fallback option)
                # If at service selection stage and no service provided, show text input for manual entry.
                if st.session_state.get("stage") == "service_selection" and st.session_state.get("service") is None:
                    input_label = "Enter your service request (e.g., appointment or payment):"
                    user_input = st.text_input(input_label, key="user_input", label_visibility="collapsed", placeholder="Type appointment, payment, or a question...")
                    if st.button("Send", key="send_service_btn"):
                        st.session_state["service"] = user_input.strip()
                        current_state = {
                            "messages": st.session_state.messages,
                            "verified": st.session_state.verified,
                            "patient": st.session_state.patient,
                            "nric": st.session_state.nric,
                            "service": st.session_state.service,
                            "stage": st.session_state.stage,
                            "last_processed_msg_idx": st.session_state.get("last_processed_msg_idx", -1),
                            "appointment_stage": st.session_state.get("appointment_stage"),
                            "current_appointment": st.session_state.get("current_appointment"),
                            "offered_slots": st.session_state.get("offered_slots"),
                            "payment_stage": st.session_state.get("payment_stage"),
                            "payment_id": st.session_state.get("payment_id"),
                            "payment_amount": st.session_state.get("payment_amount"),
                            "payment_completed": st.session_state.get("payment_completed", False),
                            "chat_started": st.session_state.get("chat_started", False)
                        }
                        result_state = workflow.invoke(current_state, config=thread_config)
                        st.session_state.update(result_state)
                        st.rerun()
                else:
                    input_label = "Enter your message:"
                    user_input = st.text_input(input_label, key="user_input", label_visibility="collapsed", placeholder="Type your message here...")
                    if st.button("Send", key="send_msg_btn"):
                        if user_input.strip():
                            st.session_state.messages.append({"role": "user", "content": user_input})
                            current_state = {
                                "messages": st.session_state.messages.copy(),
                                "verified": st.session_state.verified,
                                "patient": st.session_state.patient,
                                "nric": st.session_state.nric,
                                "service": st.session_state.service,
                                "stage": st.session_state.stage,
                                "last_processed_msg_idx": st.session_state.get("last_processed_msg_idx", -1),
                                "appointment_stage": st.session_state.get("appointment_stage"),
                                "current_appointment": st.session_state.get("current_appointment"),
                                "offered_slots": st.session_state.get("offered_slots"),
                                "payment_stage": st.session_state.get("payment_stage"),
                                "payment_id": st.session_state.get("payment_id"),
                                "payment_amount": st.session_state.get("payment_amount"),
                                "payment_completed": st.session_state.get("payment_completed", False),
                                "chat_started": st.session_state.get("chat_started", False)
                            }
                            result_state = workflow.invoke(current_state, config=thread_config)
                            st.session_state.update(result_state)
                            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Optional debug expander
        with st.expander("View Current State (Debug)"):
            st.write({
                "messages_count": len(st.session_state.messages),
                "verified": st.session_state.verified,
                "patient": st.session_state.patient,
                "nric": st.session_state.nric,
                "service": st.session_state.get("service"),
                "stage": st.session_state.get("stage"),
                "last_processed_msg_idx": st.session_state.get("last_processed_msg_idx", -1),
                "appointment_stage": st.session_state.get("appointment_stage"),
                "current_appointment": st.session_state.get("current_appointment"),
                "offered_slots": st.session_state.get("offered_slots"),
                "payment_stage": st.session_state.get("payment_stage"),
                "payment_id": st.session_state.get("payment_id"),
                "payment_amount": st.session_state.get("payment_amount"),
                "payment_completed": st.session_state.get("payment_completed", False),
                "chat_started": st.session_state.get("chat_started", False),
                "voice_mode": st.session_state.get("voice_mode", True)
            })
    
        # Always play the latest assistant message via TTS
        if st.session_state.get("messages") and st.session_state["messages"][-1]["role"] == "assistant":
            play_audio_response(st.session_state["messages"][-1]["content"])

if __name__ == "__main__":
    main()
