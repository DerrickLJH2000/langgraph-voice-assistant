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
- [Installation](#installation)
- [Running the App](#running-the-app)
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

Before setting up and running the application, ensure you have:

- **Python 3.11** installed.
- A stable internet connection (for API calls and transcription services).

### API Keys and Environment Variables

The application depends on several third-party API services. You must set up the following API keys in your Streamlit secrets file:

- **OPENAI_API_KEY:** For accessing OpenAI's Whisper transcription.
- **ELEVENLABS_API_KEY:** For text-to-speech conversion using ElevenLabs.
- **STRIPE_API_KEY:** For integrating with Stripe Checkout and payment processing.

## Installation

To install the application locally, follow these steps:

1. **Clone the Repository**

   Open your terminal and execute:

   ```bash
   git clone https://github.com/DerrickLJH2000/langgraph-voice-assistant.git
   cd clinic-voice-assistant
   ```

2. **Install Dependencies**

   Install all required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

   This command will install libraries such as Streamlit, Stripe, qrcode, Pillow, OpenAI, ElevenLabs, LangChain, and other necessary packages.

## Running the App

Once the installation is complete, you can launch your Streamlit application by running:

```bash
streamlit run app.py
```

After the server starts, your default web browser should open the app, or you can navigate to the URL provided in the terminal.

## Configuration

### Streamlit Secrets

Create or update the `streamlit/secrets.toml` file in your repository directory with the following content to securely store your API keys:

```toml
[general]
OPENAI_API_KEY = "your_openai_api_key_here"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
STRIPE_API_KEY = "your_stripe_api_key_here"
```

> **Note:** Do not share or expose your API keys publicly. This file is automatically protected by Streamlit when deployed.

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
  Verify your OpenAI API key and ensure a stable internet connection if speech-to-text functionality is not performing as expected.

- **Payment Processing Problems:**  
  Confirm your Stripe API key is correct and that you have created the required products in your Stripe dashboard. Also, check that your `products.json` file is correctly formatted.

- **Missing API Keys:**  
  Make sure you have set all necessary API keys in `streamlit/secrets.toml`. Without these, certain features (like TTS or transcription) will not work.

- **General Debugging:**  
  Use the debug expander in the app to inspect the current state, which can provide insight into problems in workflow processing.

## Contributing

Contributions are welcome! Feel free to fork this repository, make improvements, and submit a pull request. For any issues or suggestions, open an issue on the repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
