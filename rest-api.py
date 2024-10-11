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

# Define model groups for rate limiting
MODEL_GROUPS = {
    "group_1": ["llama-3.1-70b-versatile", "llama3-70b-8192"],
    "group_2": ["llama3-8b-8192", "llama-3.1-8b-instant", "llama-3.2-11b-text-preview", "llama-3.2-11b-vision-preview", "mixtral-8x7b-32768"],
    "group_3": ["llama-3.2-1b-preview", "llama-3.2-3b-preview", "llama-guard-3-8b", "llava-v1.5-7b-4096-preview", "gemma-7b-it", "gemma2-9b-it"]
}

# Rate limits for each group
RATE_LIMITS = {
    "group_1": {"requests_per_day": 28_800, "requests_per_minute": 60},
    "group_2": {"requests_per_day": 57_200, "requests_per_minute": 150},
    "group_3": {"requests_per_day": 71_600, "requests_per_minute": 180}
}

# Usage counters for rate limiting
usage_counters = {
    "group_1": {"minute": 0, "day": 0, "last_minute_reset": time.time(), "last_day_reset": time.time()},
    "group_2": {"minute": 0, "day": 0, "last_minute_reset": time.time(), "last_day_reset": time.time()},
    "group_3": {"minute": 0, "day": 0, "last_minute_reset": time.time(), "last_day_reset": time.time()}
}

# Tolerance threshold for switching groups (90%)
TOLERANCE_THRESHOLD = 0.9

# File to store rate limit counters
RATE_LIMITS_FILE = "rate_limits_counters.json"

# Load rate limit counters from file if exists
if os.path.exists(RATE_LIMITS_FILE):
    with open(RATE_LIMITS_FILE, "r") as f:
        rate_limits_counters = json.load(f)
        # Ensure timestamp values are correctly loaded
        for group in rate_limits_counters:
            rate_limits_counters[group]["last_minute_reset"] = float(rate_limits_counters[group]["last_minute_reset"])
            rate_limits_counters[group]["last_day_reset"] = float(rate_limits_counters[group]["last_day_reset"])
else:
    rate_limits_counters = {group: {"minute": 0, "day": 0, "last_minute_reset": time.time(), "last_day_reset": time.time()} for group in RATE_LIMITS}

# Function to save rate limits to file
last_written_time = time.time()

# Function to save only the necessary fields to reduce file size
def save_rate_limits():
    data_to_save = {
        group: {
            "minute": counters["minute"],
            "day": counters["day"],
            "last_minute_reset": counters["last_minute_reset"],
            "last_day_reset": counters["last_day_reset"]
        }
        for group, counters in rate_limits_counters.items()
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

def select_model():
    # Iterate through the groups in order of priority
    for group_name in ["group_1", "group_2", "group_3"]:
        rate_limits = RATE_LIMITS[group_name]
        counters = rate_limits_counters[group_name]
        current_time = time.time()

        # Reset minute and day counters if necessary
        if current_time - counters["last_minute_reset"] >= 60:
            counters["minute"] = 0
            counters["last_minute_reset"] = current_time
        if current_time - counters["last_day_reset"] >= 86400:
            counters["day"] = 0
            counters["last_day_reset"] = current_time

        # Check if usage is below the tolerance threshold
        if counters["minute"] < rate_limits["requests_per_minute"] * TOLERANCE_THRESHOLD and \
           counters["day"] < rate_limits["requests_per_day"] * TOLERANCE_THRESHOLD:
            # Select a model from this group
            for model in MODEL_GROUPS[group_name]:
                return model, group_name

    # If no group is available, raise an exception
    raise HTTPException(status_code=429, detail="Rate limit exceeded for all model groups.")

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
    model_name, group_name = select_model()

    logger.info(f"Using model: {model_name} from group: {group_name}")

    # Rate limiting checks
    current_time = time.time()
    rate_limits = RATE_LIMITS[group_name]
    counters = rate_limits_counters[group_name]

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