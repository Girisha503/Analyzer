from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Summarization models
SUMMARIZATION_MODELS = {
    "primary": "facebook/bart-large-cnn",
    "fallback": "google/pegasus-cnn_dailymail"
}

# Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS so your frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["http://127.0.0.1:5500"] for strict security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------
# ✅ Request model
class ComplaintRequest(BaseModel):
    text: str

# --------------------------------------
# ✅ Utility functions

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!?$-]', '', text)
    return text.strip()

def extract_key_info(text: str) -> dict:
    info = {}
    prices = re.findall(r'\$(\d+(?:\.\d{2})?)', text)
    if prices:
        info['prices'] = [float(p) for p in prices]
    order_nums = re.findall(r'#?([A-Z0-9-]{5,})', text)
    if order_nums:
        info['order_numbers'] = order_nums
    time_periods = re.findall(r'(\d+)\s*(days?|weeks?|months?|hours?)', text.lower())
    if time_periods:
        info['time_periods'] = time_periods
    return info

def detect_sentiment(text: str) -> str:
    text_lower = text.lower()
    anger = ['ridiculous', 'terrible', 'awful', 'angry', 'furious', 'outraged', 'unacceptable', 'disgusting']
    frustration = ['frustrated', 'disappointed', 'annoyed', 'upset', 'bothered', 'poor', 'slow', 'late', 'broken', 'damaged']
    polite = ['please', 'thank you', 'appreciate', 'understand', 'hope', 'wondering', 'grateful']
    if any(word in text_lower for word in anger):
        return "angry"
    elif any(word in text_lower for word in frustration):
        return "frustrated"
    elif any(word in text_lower for word in polite):
        return "polite"
    return "neutral"

def categorize_complaint(text: str) -> str:
    text_lower = text.lower()
    if any(word in text_lower for word in ['shipping', 'delivery', 'arrived', 'package', 'delayed']):
        return "shipping"
    elif any(word in text_lower for word in ['charge', 'billing', 'payment', 'refund', 'money']):
        return "billing"
    elif any(word in text_lower for word in ['broken', 'defective', 'quality', 'damaged', 'stopped working']):
        return "product_defect"
    elif any(word in text_lower for word in ['wrong', 'incorrect', 'mistake', 'different']):
        return "wrong_item"
    elif any(word in text_lower for word in ['customer service', 'support', 'representative', 'staff']):
        return "service"
    elif any(word in text_lower for word in ['website', 'app', 'technical', 'crash']):
        return "technical"
    elif any(word in text_lower for word in ['price', 'cost', 'pricing']):
        return "pricing"
    return "general"

def create_detailed_web_summary(text: str, key_info: dict, category: str, sentiment: str) -> str:
    summary_parts = []
    category_overviews = {
        "shipping": "Customer is experiencing shipping and delivery problems",
        "billing": "Customer has concerns about billing or payment issues",
        "product_defect": "Customer received a defective or damaged product",
        "wrong_item": "Customer received the wrong item in their order",
        "service": "Customer had a poor experience with customer service",
        "technical": "Customer is facing technical issues with the website or app",
        "pricing": "Customer has questions or concerns about pricing",
        "general": "Customer has submitted a general complaint"
    }
    sentiment_context = {
        "angry": "The customer is very upset and frustrated",
        "frustrated": "The customer is disappointed and frustrated",
        "polite": "The customer is being polite and understanding",
        "neutral": "The customer is presenting their concerns calmly"
    }
    summary_parts.append(f"{category_overviews.get(category, 'Customer complaint received')}. {sentiment_context.get(sentiment, 'Customer feedback received')}.")

    details = []
    if 'prices' in key_info:
        if len(key_info['prices']) == 1:
            details.append(f"The issue involves an amount of ${key_info['prices'][0]:.2f}")
        else:
            total = sum(key_info['prices'])
            details.append(f"The issue involves multiple amounts totaling ${total:.2f}")
    if 'order_numbers' in key_info:
        details.append(f"Order reference: {key_info['order_numbers'][0]}")
    if 'time_periods' in key_info:
        period = key_info['time_periods'][0]
        details.append(f"Timeline mentioned: {period[0]} {period[1]}")
    text_lower = text.lower()
    if 'refund' in text_lower:
        details.append("Customer is requesting a refund")
    if 'replacement' in text_lower:
        details.append("Customer wants a replacement item")
    if 'cancel' in text_lower:
        details.append("Customer wants to cancel their order")
    if 'urgent' in text_lower or 'asap' in text_lower:
        details.append("Customer indicates this is urgent")
    if 'disappointed' in text_lower or 'expected' in text_lower:
        details.append("Customer's expectations were not met")
    if details:
        summary_parts.append(f"Key details: {', '.join(details)}.")
    resolution = []
    if 'want' in text_lower:
        if 'money back' in text_lower or 'refund' in text_lower:
            resolution.append("wants their money back")
        if 'fix' in text_lower or 'resolve' in text_lower:
            resolution.append("wants the issue fixed")
        if 'speak' in text_lower or 'manager' in text_lower:
            resolution.append("wants to speak with a manager")
    if resolution:
        summary_parts.append(f"The customer {' and '.join(resolution)}.")
    priority = {
        "angry": "This requires immediate attention due to the customer's frustration level",
        "frustrated": "This should be handled promptly to prevent further frustration",
        "polite": "This can be handled through normal channels with good follow-up",
        "neutral": "This can be processed through standard procedures"
    }
    summary_parts.append(priority.get(sentiment, "Standard handling procedures apply."))
    next_steps = {
        "shipping": "Check tracking information and contact the shipping carrier",
        "billing": "Review the customer's account and billing history",
        "product_defect": "Arrange for return/replacement and check quality control",
        "wrong_item": "Send correct item and arrange pickup of wrong item",
        "service": "Have a supervisor review the service interaction",
        "technical": "Forward to technical support team for investigation",
        "pricing": "Review pricing and provide clear explanation to customer",
        "general": "Route to appropriate department for comprehensive review"
    }
    summary_parts.append(f"Recommended action: {next_steps.get(category, 'Review and respond appropriately.')}")
    return " ".join(summary_parts)

async def get_ai_summary(text: str) -> str | None:
    for _, model_id in SUMMARIZATION_MODELS.items():
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            payload = {
                "inputs": text,
                "parameters": {
                    "min_length": 100,
                    "max_length": 500,
                    "do_sample": True,
                    "length_penalty": 0.5,
                    "num_beams": 4,
                    "temperature": 0.7,
                    "repetition_penalty": 1.1
                }
            }
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                output = response.json()
                if isinstance(output, list) and output and 'summary_text' in output[0]:
                    return output[0]['summary_text']
                if isinstance(output, dict) and 'summary_text' in output:
                    return output['summary_text']
        except Exception as e:
            print(f"Error with {model_id}: {e}")
    return None

# --------------------------------------
# ✅ Endpoints

@app.post("/summarize")
async def summarize(complaint: ComplaintRequest):
    original_text = clean_text(complaint.text)
    if len(original_text.split()) < 3:
        return {"success": False, "error": "Text too short to summarize"}
    key_info = extract_key_info(original_text)
    category = categorize_complaint(original_text)
    sentiment = detect_sentiment(original_text)
    summary = create_detailed_web_summary(original_text, key_info, category, sentiment)
    ai_summary = await get_ai_summary(original_text)
    if ai_summary:
        summary += f"\n\nAI-GENERATED SUPPLEMENTARY ANALYSIS:\n{ai_summary}"
    return {
        "success": True,
        "original_text": original_text,
        "summary": summary,
        "sentiment": sentiment,
        "category": category,
        "key_info": key_info
    }

@app.post("/respond")
async def respond(complaint: ComplaintRequest):
    text = clean_text(complaint.text)
    key_info = extract_key_info(text)
    category = categorize_complaint(text)
    sentiment = detect_sentiment(text)
    if sentiment == "angry":
        start = "We sincerely apologize for your frustrating experience and understand your anger."
    elif sentiment == "frustrated":
        start = "We're sorry to hear about your disappointing experience."
    elif sentiment == "polite":
        start = "Thank you for bringing this to our attention in such a courteous manner."
    else:
        start = "Thank you for your feedback."
    category_reply = {
        "shipping": "We're investigating the shipping delay and will update you within 24 hours.",
        "billing": "We're reviewing your billing concern and will correct any errors immediately.",
        "product_defect": "We'll arrange for a replacement or refund for your defective product.",
        "wrong_item": "We'll send you the correct item and arrange pickup of the incorrect one.",
        "service": "We're addressing this service issue with our team and will follow up with you.",
        "technical": "Our technical team is working to resolve this issue as quickly as possible.",
        "pricing": "We're reviewing your pricing concern and will provide clarification.",
        "general": "We're looking into your concern and will respond with a resolution soon."
    }
    reply = category_reply.get(category, "We're looking into your concern.")
    ref = f" Reference: {key_info['order_numbers'][0]}" if 'order_numbers' in key_info else ""
    return {"success": True, "response": f"{start} {reply}{ref}"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
