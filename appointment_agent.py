# appointment_agent.py
"""
This module contains appointment-related functions for your Clinic Voice Assistant.
It handles appointment scheduling, slot selection, and related helper methods.
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict

import streamlit as st

# Import appointment retrieval and update functions from your supabase_functions module.
from supabase_functions import get_appointment_by_patient_id, update_appointment_datetime

# If not already defined in a shared module, you can define the ClinicState type here.
class ClinicState(TypedDict):
    messages: List
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


def extract_number_from_text(text: str) -> int:
    """
    Extract a number from a given text string.
    Handles both direct digit matches and small number words.
    """
    if not text:
        return -1

    text = text.strip().lower()
    # First, try matching common number words.
    simple_words = {
        'one': 1, 'two': 2, 'three': 3,
        'first': 1, 'second': 2, 'third': 3
    }
    word_pattern = r'\b(' + '|'.join(simple_words.keys()) + r')\b'
    word_match = re.search(word_pattern, text)
    if word_match:
        return simple_words[word_match.group(1)]

    # Try a direct digit match.
    digit_match = re.search(r'\b([1-9])\b', text)
    if digit_match:
        return int(digit_match.group(1))

    # Try matching expressions like "option 1", "slot two", etc.
    pattern_match = re.search(r'(?:option|choice|number|slot)\s*#?\s*([1-9])', text)
    if pattern_match:
        return int(pattern_match.group(1))

    # As a fallback, check for ordinal expressions.
    ordinal_match = re.search(
        r'(?:the|choose|select|pick)\s+(?:([1-9])(?:st|nd|rd|th)|(?:first|second|third))',
        text
    )
    if ordinal_match and ordinal_match.group(1):
        return int(ordinal_match.group(1))
    elif "first" in text:
        return 1
    elif "second" in text:
        return 2
    elif "third" in text:
        return 3

    # If nothing is found, return -1.
    return -1


def debug_extract_number_from_text(text: str) -> int:
    """
    Debug version of extract_number_from_text that logs intermediate steps.
    """
    print(f"Attempting to extract number from: '{text}'")
    if not text:
        print("Empty input text")
        return -1

    text = text.strip().lower()
    print(f"Normalized text: '{text}'")

    simple_words = {
        'one': 1, 'two': 2, 'three': 3,
        'first': 1, 'second': 2, 'third': 3
    }
    word_pattern = r'\b(' + '|'.join(simple_words.keys()) + r')\b'
    print(f"Checking for word pattern: {word_pattern}")
    word_match = re.search(word_pattern, text)
    if word_match:
        result = simple_words[word_match.group(1)]
        print(f"Found word match: '{word_match.group(1)}' -> {result}")
        return result

    print("Checking for direct digit match")
    digit_match = re.search(r'\b([1-9])\b', text)
    if digit_match:
        result = int(digit_match.group(1))
        print(f"Found digit match: '{digit_match.group(1)}' -> {result}")
        return result

    print("No matching patterns found, returning -1")
    return -1


def process_appointment_selection(state: ClinicState) -> ClinicState:
    """
    Process the appointment slot selection based on the user's input.
    """
    last_user_message = None
    for msg in reversed(state["messages"]):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break

    if not last_user_message:
        state["messages"].append(
            {"role": "assistant", "content": "I'm waiting for your input regarding the appointment slot selection."}
        )
        return state

    if state.get("appointment_stage") == "selecting_slot":
        # Use debug version to inspect how the number is extracted.
        slot_number = debug_extract_number_from_text(last_user_message)
        if 1 <= slot_number <= 3:
            slot_index = slot_number - 1
            if not state.get("offered_slots") or slot_index >= len(state.get("offered_slots", [])):
                state["messages"].append(
                    {"role": "assistant", "content": "There was an issue with your selection. Please try again."}
                )
                return state

            selected_slot = state["offered_slots"][slot_index]
            if "current_appointment" in state:
                appointment_id = state["current_appointment"]["appointment_id"]
                success = update_appointment_datetime(
                    appointment_id, selected_slot["date"], selected_slot["time"]
                )
                if success:
                    state["messages"].append(
                        {"role": "assistant", "content": f"Your appointment has been rescheduled to {selected_slot['date']} at {selected_slot['time']}."}
                    )
                else:
                    state["messages"].append(
                        {"role": "assistant", "content": "I wasn't able to update your appointment. Please try again later."}
                    )
            # Clear appointment-related state.
            state["appointment_stage"] = None
            if "offered_slots" in state: del state["offered_slots"]
            if "current_appointment" in state: del state["current_appointment"]
        else:
            state["messages"].append({
                "role": "assistant",
                "content": "I didn't recognize that as a slot selection. Please choose one of the provided options by saying 'one', 'two', or 'three', or the corresponding number."
            })

    return state


def appointment_agent_node(state: ClinicState) -> ClinicState:
    """
    Appointment agent node that either processes a slot selection or presents available appointment options.
    """
    # If in slot-selection stage, try to process the selection.
    if state.get("appointment_stage") == "selecting_slot":
        last_user_message = None
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        if last_user_message:
            slot_number = extract_number_from_text(last_user_message)
            if 1 <= slot_number <= 3:
                slot_index = slot_number - 1
                if state.get("offered_slots") and slot_index < len(state.get("offered_slots", [])):
                    selected_slot = state["offered_slots"][slot_index]
                    if "current_appointment" in state:
                        appointment_id = state["current_appointment"]["appointment_id"]
                        success = update_appointment_datetime(
                            appointment_id, selected_slot["date"], selected_slot["time"]
                        )
                        if success:
                            state["messages"].append(
                                {"role": "assistant", "content": f"Your appointment has been rescheduled to {selected_slot['date']} at {selected_slot['time']}."}
                            )
                        else:
                            state["messages"].append(
                                {"role": "assistant", "content": "I wasn't able to update your appointment. Please try again later."}
                            )
                    # Clear the appointment stage and related data.
                    state["appointment_stage"] = None
                    if "offered_slots" in state: del state["offered_slots"]
                    if "current_appointment" in state: del state["current_appointment"]
                    return state
                else:
                    state["messages"].append(
                        {"role": "assistant", "content": "There was an issue with your selection. Please try again."}
                    )
                    return state
            elif last_user_message.strip() and not any(
                msg["content"].startswith("I didn't recognize") for msg in reversed(state["messages"][:5]) if msg["role"] == "assistant"
            ):
                state["messages"].append({
                    "role": "assistant",
                    "content": "I didn't recognize that as a slot selection. Please choose one of the provided options by saying 'one', 'two', or 'three', or the corresponding number."
                })
                return state

    # If not in a selection stage, present available appointment options.
    patient = state.get("patient", {})
    patient_id = patient.get("patient_id") if patient else None
    if not patient_id:
        state["messages"].append(
            {"role": "assistant", "content": "I need your patient information to schedule appointments."}
        )
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

        # Define three possible time slots.
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
        state["messages"].append(
            {"role": "assistant", "content": "You have no upcoming appointments. Restarting to the main page..."}
        )
        time.sleep(5)
        # Clear Streamlit session state and rerun.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    return state
