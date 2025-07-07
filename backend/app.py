from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# Load your OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Data structure for incoming JSON
class Complaint(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Backend is working!"}

@app.post("/summarize")
def summarize(complaint: Complaint):
    # Validate input
    if not complaint.text.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    try:
        # Call OpenAI for summarization
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer service agent. Summarize customer complaints clearly and briefly."},
                {"role": "user", "content": f"Complaint: {complaint.text}"}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")



@app.post("/generate-response")
def generate_response(complaint: Complaint):
    # Validate input
    if not complaint.text.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    try:
        # Call OpenAI to generate a polite reply
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a polite, professional customer support agent. Write a helpful and empathetic response to the customer's complaint."},
                {"role": "user", "content": f"Complaint: {complaint.text}"}
            ]
        )
        reply = response.choices[0].message.content.strip()
        return {"response": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during response generation: {str(e)}")

