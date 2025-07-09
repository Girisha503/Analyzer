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

def simple_sentence_split(text):
    """Simple sentence splitting without NLTK."""
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_key_info(text):
    """Extract key information like prices, dates, order numbers."""
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

def detect_sentiment(text):
    """Rule-based sentiment detection."""
    text_lower = text.lower()
    
    # Enhanced rule-based sentiment
    anger_words = ['ridiculous', 'terrible', 'awful', 'angry', 'furious', 'outraged', 'unacceptable', 'disgusting']
    frustration_words = ['frustrated', 'disappointed', 'annoyed', 'upset', 'bothered', 'poor', 'slow', 'late', 'broken', 'damaged']
    polite_words = ['please', 'thank you', 'appreciate', 'understand', 'hope', 'wondering', 'grateful']
    
    if any(word in text_lower for word in anger_words):
        return "angry"
    elif any(word in text_lower for word in frustration_words):
        return "frustrated"
    elif any(word in text_lower for word in polite_words):
        return "polite"
    return "neutral"

def categorize_complaint(text):
    """Categorize the complaint type."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['shipping', 'delivery', 'arrived', 'package', 'delayed']):
        return "shipping"
    elif any(word in text_lower for word in ['charge', 'billing', 'payment', 'refund', 'money']):
        return "billing"
    elif any(word in text_lower for word in ['broken', 'defective', 'quality', 'damaged', 'stopped working']):
        return "product_defect"
    
    # Wrong item
    elif any(word in text_lower for word in ['wrong', 'incorrect', 'mistake', 'different', 'not what i ordered']):
        return "wrong_item"
    
    # Shipping issues
    elif any(word in text_lower for word in ['shipping', 'delivery', 'arrived', 'package', 'delayed', 'tracking', 'late delivery']):
        return "shipping"
    
    # Billing issues
    elif any(word in text_lower for word in ['charge', 'billing', 'payment', 'refund', 'money', 'charged', 'overcharged']):
        return "billing"
    
    # Customer service
    elif any(word in text_lower for word in ['customer service', 'support', 'representative', 'staff', 'rude', 'unhelpful']):
        return "service"
    
    # Technical issues
    elif any(word in text_lower for word in ['website', 'app', 'technical', 'crash', 'login', 'error']):
        return "technical"
    
    # Pricing concerns
    elif any(word in text_lower for word in ['price', 'cost', 'pricing', 'expensive', 'overpriced']):
        return "pricing"
    else:
        return "general"

def create_detailed_web_summary(text, key_info, category, sentiment):
    """Create a detailed, organized web-friendly summary."""
    
    text_lower = text.lower()
    
    # Extract specific details from text
    product_mentioned = "item"
    if "laptop" in text_lower:
        product_mentioned = "laptop"
    elif "phone" in text_lower:
        product_mentioned = "phone"
    elif "tablet" in text_lower:
        product_mentioned = "tablet"
    elif "computer" in text_lower:
        product_mentioned = "computer"
    elif "product" in text_lower:
        product_mentioned = "product"
    
    # Build main issue description
    if category == "product_defect":
        if "damaged" in text_lower:
            main_issue = f"The customer reports that their **{product_mentioned}** arrived in a **damaged condition**."
        elif "defective" in text_lower:
            main_issue = f"The customer reports that their **{product_mentioned}** is **defective** and not functioning properly."
        elif "broken" in text_lower:
            main_issue = f"The customer reports that their **{product_mentioned}** is **broken** and unusable."
        else:
            main_issue = f"The customer reports quality issues with their **{product_mentioned}**."
    elif category == "shipping":
        main_issue = f"The customer reports that their **{product_mentioned}** order has not arrived as expected."
    elif category == "billing":
        main_issue = f"The customer has raised concerns about **billing or payment issues** related to their purchase."
    elif category == "wrong_item":
        main_issue = f"The customer reports receiving the **wrong item** instead of their ordered **{product_mentioned}**."
    elif category == "service":
        main_issue = f"The customer reports a **poor customer service experience** regarding their **{product_mentioned}**."
    else:
        main_issue = f"The customer has reported concerns with their **{product_mentioned}**."
    
    # Add time and order details if available
    details = []
    if 'time_periods' in key_info and key_info['time_periods']:
        time_detail = key_info['time_periods'][0]
        details.append(f"approximately **{time_detail[0]} {time_detail[1]} ago**")
    
    if 'order_numbers' in key_info and key_info['order_numbers']:
        details.append(f"The order reference number is **{key_info['order_numbers'][0]}**")
    
    if 'prices' in key_info and key_info['prices']:
        if len(key_info['prices']) == 1:
            details.append(f"the purchase amount is **${key_info['prices'][0]:.2f}**")
        else:
            total = sum(key_info['prices'])
            details.append(f"the total purchase amount is **${total:.2f}**")
    
    # Add details to main issue
    if details:
        main_issue += f" {' and '.join(details)}."
    
    # Customer sentiment and additional concerns
    sentiment_descriptions = {
        "angry": "The customer expresses **strong anger and frustration** with emphatic language, indicating high dissatisfaction",
        "frustrated": "The customer shows **disappointment and frustration** regarding the situation",
        "polite": "The customer maintains a **courteous and respectful tone** while expressing their concerns",
        "neutral": "The customer presents their concerns in a **calm and factual manner**"
    }
    
    sentiment_text = sentiment_descriptions.get(sentiment, "The customer has expressed their concerns")
    
    # Check for additional service issues
    service_issues = []
    if "unhelpful" in text_lower:
        service_issues.append("unhelpful support staff")
    if "rude" in text_lower:
        service_issues.append("rude customer service")
    if "poor service" in text_lower:
        service_issues.append("poor service quality")
    
    additional_paragraph = sentiment_text
    if service_issues:
        additional_paragraph += f", and also mentions issues with **{' and '.join(service_issues)}**"
    additional_paragraph += "."
    
    # Resolution requests and urgency
    resolution_requests = []
    if "refund" in text_lower:
        resolution_requests.append("**full refund**")
    if "replacement" in text_lower or "replace" in text_lower:
        resolution_requests.append("**replacement item**")
    if "fix" in text_lower or "repair" in text_lower:
        resolution_requests.append("**repair or fix**")
    if "investigate" in text_lower:
        resolution_requests.append("**investigation into the issue**")
    if "update" in text_lower:
        resolution_requests.append("**status update**")
    
    # Determine urgency
    urgency = "Standard"
    if "urgent" in text_lower or "immediately" in text_lower or "asap" in text_lower:
        urgency = "High — immediate attention required"
    elif "soon" in text_lower or "quickly" in text_lower:
        urgency = "Medium — prompt response expected"
    else:
        urgency = "Standard — normal processing timeframe"
    
    resolution_paragraph = ""
    if resolution_requests:
        resolution_paragraph += f"The customer specifically requests {', '.join(resolution_requests)}. "
    
    resolution_paragraph += f"The complaint indicates **{urgency.split(' — ')[0].lower()} priority** with expectations for **{urgency.split(' — ')[1] if ' — ' in urgency else 'standard processing'}**."
    
    # Key details section
    key_details = []
    
    if 'order_numbers' in key_info and key_info['order_numbers']:
        key_details.append(f"**Order Number:** {key_info['order_numbers'][0]}")
    
    key_details.append(f"**Product:** {product_mentioned.title()}")
    
    if 'prices' in key_info and key_info['prices']:
        if len(key_info['prices']) == 1:
            key_details.append(f"**Purchase Amount:** ${key_info['prices'][0]:.2f}")
        else:
            total = sum(key_info['prices'])
            key_details.append(f"**Total Amount:** ${total:.2f}")
    
    # Category display mapping
    category_display = {
        "shipping": "Shipping Delay",
        "billing": "Billing Issue", 
        "product_defect": "Product Defect",
        "wrong_item": "Wrong Item",
        "service": "Customer Service",
        "technical": "Technical Issue",
        "pricing": "Pricing Concern",
        "general": "General Inquiry"
    }
    
    key_details.append(f"**Issue Category:** {category_display.get(category, 'General')}")
    
    # Sentiment display
    sentiment_display = {
        "angry": "Angry, Frustrated",
        "frustrated": "Disappointed, Frustrated", 
        "polite": "Courteous, Understanding",
        "neutral": "Neutral, Calm"
    }
    
    key_details.append(f"**Sentiment:** {sentiment_display.get(sentiment, 'Neutral')}")
    key_details.append(f"**Urgency:** {urgency.split(' — ')[0]}")
    
    # Combine everything
    summary = f"{main_issue}\n\n{additional_paragraph}\n\n{resolution_paragraph}\n\n"
    summary += "**Key Details Extracted:**\n"
    summary += "\n".join([f"* {detail}" for detail in key_details])
    
    return summary

async def get_ai_summary(text):
    """Get AI-generated summary with improved formatting."""
    
    for model_name, model_id in SUMMARIZATION_MODELS.items():
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            # Improved prompt for better analysis
            prompt_text = f"Analyze this customer complaint professionally, focusing on the main issue, customer sentiment, and resolution needed: {text}"
            
            payload = {
                "inputs": prompt_text,
                "parameters": {
                    "min_length": 50,
                    "max_length": 200,
                    "do_sample": True,
                    "length_penalty": 1.0,
                    "num_beams": 4,
                    "temperature": 0.4,
                    "repetition_penalty": 1.3,
                    "no_repeat_ngram_size": 3
                }
            }
            
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                output = response.json()
                
                if isinstance(output, list) and len(output) > 0:
                    if 'summary_text' in output[0]:
                        return clean_text(output[0]['summary_text'])
                elif isinstance(output, dict):
                    if 'summary_text' in output:
                        return clean_text(output['summary_text'])
                
        except Exception as e:
            print(f"Error with {model_id}: {e}")
    
    return None

@app.post("/summarize")
async def summarize(complaint: ComplaintRequest):
    original_text = clean_text(complaint.text)
    
    if len(original_text.split()) < 3:
        return {"success": False, "error": "Text too short to summarize"}
    
    # Extract information
    key_info = extract_key_info(original_text)
    category = categorize_complaint(original_text)
    sentiment = detect_sentiment(original_text)
    
    # Create organized summary
    summary = create_detailed_web_summary(original_text, key_info, category, sentiment)
    
    # Get AI analysis as supplementary information
    ai_summary = await get_ai_summary(original_text)
    
    response_data = {
        "success": True,
        "original_text": original_text,
        "summary": summary,
        "sentiment": sentiment,
        "category": category,
        "key_info": key_info,
        "summary_word_count": len(summary.split())
    }
    
    # Add AI analysis if available
    if ai_summary:
        response_data["ai_analysis"] = f"**AI-Generated Supplementary Analysis:**\n{ai_summary}"
    
    return response_data

@app.post("/respond")
async def respond(complaint: ComplaintRequest):
    cleaned = clean_text(complaint.text)
    
    key_info = extract_key_info(cleaned)
    category = categorize_complaint(cleaned)
    sentiment = detect_sentiment(cleaned)
    
    if sentiment == "angry":
        response_start = "We sincerely apologize for your frustrating experience and understand your anger."
    elif sentiment == "frustrated":
        response_start = "We're sorry to hear about your disappointing experience."
    elif sentiment == "polite":
        response_start = "Thank you for bringing this to our attention in such a courteous manner."
    else:
        response_start = "Thank you for your feedback."
    
    category_responses = {
        "shipping": "We're investigating the shipping delay and will provide you with an update within 24 hours.",
        "billing": "We're reviewing your billing concern and will correct any errors immediately.",
        "product_defect": "We'll arrange for a replacement or refund for your defective product.",
        "wrong_item": "We'll send you the correct item and arrange pickup of the incorrect one.",
        "service": "We're addressing this service issue with our team and will follow up with you.",
        "technical": "Our technical team is working to resolve this issue as quickly as possible.",
        "pricing": "We're reviewing your pricing concern and will provide clarification.",
        "general": "We're looking into your concern and will respond with a resolution soon."
    }
    
    category_response = category_responses.get(category, "We're looking into your concern.")
    
    reference = ""
    if 'order_numbers' in key_info:
        reference = f" Reference: {key_info['order_numbers'][0]}"
    
    full_response = f"{response_start} {category_response}{reference}"
    
    return {
        "success": True,
        "response": full_response,
        "sentiment": sentiment,
        "category": category
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)