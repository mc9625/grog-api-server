from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a default model
DEFAULT_MODEL = "llama3-70b-8192"

# Define the request body model
class RequestBody(BaseModel):
    text: str
    auth_key: str
    option: dict | None = None

# Initialize FastAPI app
app = FastAPI()

@app.post("/custom-llm/")
async def custom_llm(request_body: RequestBody):
    # Log the received request body for debugging purposes
    #logger.info(f"Received request body: {request_body}")

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

    #logger.info(f"Using model: {model_name}")

    try:
        # Initialize Groq client with the provided auth_key
        groq_client = Groq(api_key=auth_key)

        # Send the request to the Groq API
        #logger.info(f"Sending request to Groq API with model {model_name}")
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": user_message}],
            model=model_name,
            max_tokens=7000
        )
        
        # Extract the assistant's response
        assistant_message = response.choices[0].message.content
        #logger.info(f"Received response from Groq: {assistant_message[:50]}...")
        
        return {"text": assistant_message}
    except Exception as e:
        # Log the error and raise HTTP exception
        logger.error(f"Error during Groq API call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during Groq API call: {str(e)}")

# Run the application if the script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
