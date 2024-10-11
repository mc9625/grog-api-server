from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
import uvicorn
import logging
import time
import json
import os
from typing import Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a default model
DEFAULT_MODEL = "llama3-70b-8192"

# Define rate limits for each model
RATE_LIMITS = {
    "gemma-7b-it": {"requests_per_minute": 30, "requests_per_day": 14400},
    "gemma2-9b-it": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llama-3.1-70b-versatile": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llama-3.1-8b-instant": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llama-3.2-11b-text-preview": {"requests_per_minute": 30, "requests_per_day": 7000},
    "llama-3.2-11b-vision-preview": {"requests_per_minute": 30, "requests_per_day": 7000},
    "llama-3.2-1b-preview": {"requests_per_minute": 30, "requests_per_day": 7000},
    "llama-3.2-3b-preview": {"requests_per_minute": 30, "requests_per_day": 7000},
    "llama-3.2-90b-text-preview": {"requests_per_minute": 30, "requests_per_day": 7000},
    "llama-guard-3-8b": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llama3-70b-8192": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llama3-8b-8192": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llama3-groq-70b-8192-tool-use-preview": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llama3-groq-8b-8192-tool-use-preview": {"requests_per_minute": 30, "requests_per_day": 14400},
    "llava-v1.5-7b-4096-preview": {"requests_per_minute": 30, "requests_per_day": 14400},
    "mixtral-8x7b-32768": {"requests_per_minute": 30, "requests_per_day": 14400}
}

# File to store rate limit counters
RATE_LIMITS_FILE = "rate_limits_counters.json"

# Load rate limit counters from file if exists
if os.path.exists(RATE_LIMITS_FILE):
    with open(RATE_LIMITS_FILE, "r") as f:
        rate_limits_counters = json.load(f)
        # Ensure timestamp values are correctly loaded
        for model in rate_limits_counters:
            rate_limits_counters[model]["last_minute_reset"] = float(rate_limits_counters[model]["last_minute_reset"])
            rate_limits_counters[model]["last_day_reset"] = float(rate_limits_counters[model]["last_day_reset"])
else:
    rate_limits_counters = {model: {"minute": 0, "day": 0, "last_minute_reset": time.time(), "last_day_reset": time.time()} for model in RATE_LIMITS}

# Function to save rate limits to file
last_written_time = time.time()

# Function to save only the necessary fields to reduce file size
def save_rate_limits():
    data_to_save = {
        model: {
            "minute": counters["minute"],
            "day": counters["day"],
            "last_minute_reset": counters["last_minute_reset"],
            "last_day_reset": counters["last_day_reset"]
        }
        for model, counters in rate_limits_counters.items()
    }
    with open(RATE_LIMITS_FILE, "w") as f:
        json.dump(data_to_save, f)

# Define the request body model
class RequestBody(BaseModel):
    text: str
    auth_key: str
    option: Union[dict, None] = None

# Initialize FastAPI app
app = FastAPI()

@app.post("/custom-llm/")
async def custom_llm(request_body: RequestBody):
    # Log the received request body for debugging purposes
    logger.info(f"Received request body: {request_body}")

    user_message = request_body.text
    auth_key = request_body.auth_key
    options = request_body.option

    # Validate that auth_key is provided
    if not auth_key:
        raise HTTPException(status_code=400, detail="auth_key is required")

    # Extract model name from option or use default
    model_name = DEFAULT_MODEL
    if options and isinstance(options, dict):
        model_name = options.get("model", DEFAULT_MODEL)

    logger.info(f"Using model: {model_name}")

    # Rate limiting checks
    current_time = time.time()
    rate_limits = RATE_LIMITS.get(model_name, None)
    counters = rate_limits_counters[model_name]

    # Reset counters if needed
    if current_time - counters["last_minute_reset"] >= 60:
        counters["minute"] = 0
        counters["last_minute_reset"] = current_time
    if current_time - counters["last_day_reset"] >= 86400:
        counters["day"] = 0
        counters["last_day_reset"] = current_time

    # Check rate limits
    if counters["minute"] >= rate_limits["requests_per_minute"]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded: too many requests per minute.")
    if counters["day"] >= rate_limits["requests_per_day"]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded: too many requests per day.")

    # Increment counters
    counters["minute"] += 1
    counters["day"] += 1

    # Save rate limits to file every minute to reduce frequency of writes
    global last_written_time
    if current_time - last_written_time >= 60:
        save_rate_limits()
        last_written_time = current_time

    try:
        # Initialize Groq client with the provided auth_key
        groq_client = Groq(api_key=auth_key)

        # Send the request to the Groq API
        logger.info(f"Sending request to Groq API with model {model_name}")
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": user_message}],
            model=model_name,
            max_tokens=7000
        )
        
        # Extract the assistant's response
        assistant_message = response.choices[0].message.content
        logger.info(f"Received response from Groq: {assistant_message[:50]}...")
        
        return {"text": assistant_message}
    except Exception as e:
        # Log the error and raise HTTP exception
        logger.error(f"Error during Groq API call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during Groq API call: {str(e)}")

# Run the application if the script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
