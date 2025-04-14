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
- [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [Configuration](#configuration)
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

Before deploying the application, ensure you have:

- A [Streamlit Cloud](https://streamlit.io/cloud) account.
- A stable internet connection for API calls and transcription services.
- Valid API keys for the required services (see [Configuration](#configuration) below).

## Deployment on Streamlit Cloud

Deploying on Streamlit Cloud is straightforward:

1. **Push to GitHub:**
   - Ensure your project is committed to a GitHub repository.

2. **Create a New App on Streamlit Cloud:**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
   - Click on **"New app"**.
   - Select your GitHub repository and set the branch and main file (e.g., `app.py`).

3. **Configure Secrets:**
   - Streamlit Cloud provides a way to securely store sensitive credentials. In your app’s dashboard, click on **"Secrets"**.
   - Add the following secrets:

     ```toml
     [general]
     OPENAI_API_KEY = "your_openai_api_key_here"
     ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
     STRIPE_API_KEY = "your_stripe_api_key_here"
     ```

     These values will automatically be loaded into your application and can be accessed via `st.secrets`.

4. **Deploy:**
   - After configuring your secrets, click **"Deploy"**. Your app will be built and hosted on Streamlit Cloud.

## Configuration

### API Keys and Environment Variables

The application depends on several third-party API services. Instead of cloning and running locally, you’ll configure your keys directly on Streamlit Cloud using its secrets management.

The following API keys must be provided:

- **OPENAI_API_KEY:** Access OpenAI's Whisper transcription service.
- **ELEVENLABS_API_KEY:** Enable text-to-speech conversion using ElevenLabs.
- **STRIPE_API_KEY:** Integrate with Stripe Checkout for payment processing.

These keys are referenced in the app’s code and can be saved in the Streamlit secrets file as shown above. Additionally, the app features a fixed sidebar where users can enter their own OpenAI and ElevenLabs keys to override the stored values if desired.

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


Ensure that the resources inside the `utils` folder are correctly placed for the application to locate product details and logos.

## Troubleshooting

- **Voice Transcription Issues:**  
  Verify that your OpenAI API key is valid and that your internet connection is stable if speech-to-text functionality is not performing as expected.

- **Payment Processing Problems:**  
  Confirm your Stripe API key is correct, ensure you have created the required products in your Stripe dashboard, and verify the `products.json` file is correctly formatted.

- **Missing API Keys:**  
  Ensure you have set all necessary API keys in your Streamlit secrets on the Cloud dashboard. Without these, certain features (like TTS or transcription) will not work.

- **General Debugging:**  
  Use the debug expander in the app to inspect the current state, which can provide insight into problems in workflow processing.