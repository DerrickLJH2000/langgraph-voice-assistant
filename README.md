# Clinic Voice Assistant

The Clinic Voice Assistant is a Streamlit-based application designed to help patients interact with their clinic through voice and text. The app supports functionalities such as:

- **NRIC Verification:** Confirms user identity using NRIC.
- **Appointment Scheduling:** Allows users to view, reschedule, or create appointments.
- **Payment Processing:** Integrates with Stripe for handling payments and displays a QR code for easy transactions.
- **FAQ Assistance:** Answers frequently asked questions about the clinic.
- **Audio Interaction:** Uses OpenAI Whisper for speech-to-text conversion and ElevenLabs for text-to-speech responses.
- **Conversational Flow:** Implements state management with LangGraph for a fluid and dynamic chat-based user experience.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Accessing the App on Streamlit Cloud](#accessing-the-app-on-streamlit-cloud)
- [Configuration Within the App](#configuration-within-the-app)
- [Folder Structure](#folder-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Voice and Text Input:** Toggle between voice-based input and traditional text messaging.
- **Stripe Integration:** Secure payment processing with real-time status updates.
- **Real-Time Transcription:** Uses OpenAI’s Whisper model to transcribe speech.
- **Text-to-Speech Responses:** Converts assistant messages to speech using ElevenLabs.
- **Appointment, Payment & FAQ Agents:** Differentiates between appointment scheduling, payments, and FAQ queries to provide tailored responses.
- **State Management:** Employs LangGraph and custom nodes for handling multi-stage interactions.

## Prerequisites

Before using the application, ensure you have a stable internet connection for API calls and transcription services, as the app depends on various third-party API services.

## Accessing the App on Streamlit Cloud

You can access the Clinic Voice Assistant directly on Streamlit Cloud without needing to configure any secrets file. Simply visit:

[https://langgraph-voice-assistant.streamlit.app/](https://langgraph-voice-assistant.streamlit.app/)

Upon visiting the app, you will be prompted in the sidebar to input your own API keys, so that you can use the app's full functionality.

## Configuration Within the App

The application is designed so that you don’t have to set up any secrets file manually. Instead:

- The sidebar includes fields where you can enter your **OpenAI API Key** and **ElevenLabs API Key**.
- These API keys will be stored in the session state for the duration of your session.
- Once entered, the app uses these keys for functionalities like transcription (using OpenAI Whisper) and text-to-speech (using ElevenLabs).

> **Note:** For Stripe payment processing, you can have the default configuration if payments are enabled, or the application may also allow you to update keys from the app interface depending on how the payment logic is set up.

## Folder Structure

A typical folder structure for the project may look like this:
```
clinic-voice-assistant/
├── app.py                   # Main Streamlit application file
├── README.md                # Project documentation (this file)
├── requirements.txt         # List of Python dependencies
├── streamlit/
│   └── secrets.toml         # Configuration file with API keys
├── utils/
│   ├── products.json        # Product data for payment processing
│   └── vocare_logo.PNG      # Logo used in QR code generation
├── supabase_functions.py    # Module with functions to interact with Supabase
└── appointment_agent.py     # Handles appointment logic and scheduling
```

## Troubleshooting

- **Voice Transcription Issues:**  
  Verify that your OpenAI API key is valid and that your internet connection is stable if the speech-to-text functionality is not performing as expected.

- **Payment Processing Problems:**  
  Confirm your Stripe API key is correct, ensure you have created the required products in your Stripe dashboard, and verify that the `products.json` file is correctly formatted.

- **Missing API Keys:**  
  Make sure to enter your OpenAI and ElevenLabs API keys using the sidebar within the app. Without these, certain features (like transcription and text-to-speech) will not work.

- **General Debugging:**  
  Use the debug expander in the app to inspect the current state, which can provide insight into problems in workflow processing.
