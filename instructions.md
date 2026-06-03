# Estigia Chatbot Web UI Instructions

We have created a beautiful web interface for you to interact with the Estigia CubeSat model locally, avoiding the plain terminal. We've used **Streamlit**, which creates modern chat interfaces directly from Python scripts.

## Setup Requirements

1. Ensure your Python environment is active.
2. Install the necessary requirements (we've added `streamlit` to your `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the Ollama server is running locally with the expected model `franciscobdl/Estigia2:latest`. 

## How to Launch the Web App

From your terminal in the `EstigiaChatbot` folder, run the following command:

```bash
streamlit run streamlit_app.py
```

## How to Use

- A new tab will automatically open in your default web browser (typically at http://localhost:8501).
- In the sidebar on the left, you can **change the language** dynamically between Español, Valencià, and English.
- Use the **chat input box** at the bottom to send queries to Estigia or ask for telemetry data.
- The web app automatically switches between providing Telemetry data via `joblib` (like "temp", "current") or using `Ollama` for general conversation, just like your terminal script.
- The chat history is preserved until you change the language or refresh the page.

Enjoy chatting with Estigia! 🛰️
