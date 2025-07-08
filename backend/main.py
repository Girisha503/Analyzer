from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import re

# Load environment vars
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Better models for summarization
SUMMARIZATION_MODELS = {
    "primary": "facebook/bart-large-cnn",
    "fallback": "google/pegasus-cnn_dailymail"
}

app = FastAPI()

class ComplaintRequest(BaseModel):
    text: str

def clean_text(text):
    """Remove extra whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!?$-]', '', text)
    return text.strip()

def simple_sentence_split(text):
    """Simple sentence splitting without NLTK."""
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_key_info(text):
    """Extract key information like prices, dates, order numbers."""
    info = {}
    
    # Extract prices
    prices = re.findall(r'\$(\d+(?:\.\d{2})?)', text)
    if prices:
        info['prices'] = [float(p) for p in prices]
    
    # Extract order numbers
    order_nums = re.findall(r'#?([A-Z0-9-]{5,})', text)
    if order_nums:
        info['order_numbers'] = order_nums
    
    # Extract time periods
    time_periods = re.findall(r'(\d+)\s*(days?|weeks?|months?|hours?)', text.lower())
    if time_periods:
        info['time_periods'] = time_periods
    
    return info

def detect_sentiment(text):
    """Enhanced rule-based sentiment detection."""
    text_lower = text.lower()
    
    # Enhanced sentiment words
    anger_words = ['ridiculous', 'terrible', 'awful', 'angry', 'furious', 'outraged', 'unacceptable', 'disgusting', 'fed up', 'sick of', 'horrible', 'worst']
    frustration_words = ['frustrated', 'disappointed', 'annoyed', 'upset', 'bothered', 'poor', 'slow', 'late', 'broken', 'damaged', 'desperate', 'tired of', 'concerned']
    polite_words = ['please', 'thank you', 'appreciate', 'understand', 'hope', 'wondering', 'grateful', 'kindly', 'would like']
    
    # Count sentiment indicators
    anger_count = sum(1 for word in anger_words if word in text_lower)
    frustration_count = sum(1 for word in frustration_words if word in text_lower)
    polite_count = sum(1 for word in polite_words if word in text_lower)
    
    # Determine sentiment based on strongest indicators
    if anger_count >= 2 or any(word in text_lower for word in ['ridiculous', 'terrible', 'awful', 'furious', 'outraged', 'unacceptable']):
        return "angry"
    elif frustration_count >= 2 or anger_count >= 1:
        return "frustrated"
    elif polite_count >= 2:
        return "polite"
    else:
        return "neutral"

def categorize_complaint(text):
    """Improved categorization logic."""
    text_lower = text.lower()
    
    # Product defect indicators (check first as they're most specific)
    if any(word in text_lower for word in ['broken', 'defective', 'damaged', 'faulty', 'cracked', 'not working', 'stopped working', 'malfunctioning']):
        return "product_defect"
    
    # Quality issues
    elif any(word in text_lower for word in ['poor quality', 'cheaply made', 'flimsy', 'cheap quality']):
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

def detect_urgency(text):
    """Detect if the complaint indicates urgency."""
    text_lower = text.lower()
    urgency_words = ['urgent', 'immediately', 'asap', 'emergency', 'right now', 'critical', 'desperate', 'need help now']
    return any(word in text_lower for word in urgency_words)

def detect_product_mentions(text):
    """Extract product mentions from the text."""
    text_lower = text.lower()
    products = []
    
    # Common product categories
    product_keywords = {
        'laptop': ['laptop', 'computer', 'notebook'],
        'phone': ['phone', 'mobile', 'smartphone', 'iphone', 'android'],
        'tablet': ['tablet', 'ipad'],
        'headphones': ['headphones', 'earbuds', 'earphones'],
        'camera': ['camera', 'camcorder'],
        'tv': ['tv', 'television', 'monitor'],
        'gaming': ['gaming', 'console', 'xbox', 'playstation', 'nintendo'],
        'appliance': ['refrigerator', 'washing machine', 'dishwasher', 'microwave']
    }
    
    for product_type, keywords in product_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            products.append(product_type)
    
    return products

def extract_specific_issues(text):
    """Extract specific issues mentioned in the complaint."""
    text_lower = text.lower()
    issues = []
    
    # Product issues
    if any(word in text_lower for word in ['broken', 'cracked', 'damaged']):
        issues.append("physical_damage")
    
    if any(word in text_lower for word in ['not working', 'stopped working', 'defective']):
        issues.append("functional_defect")
    
    if 'wrong item' in text_lower or 'incorrect' in text_lower:
        issues.append("wrong_item")
    
    # Service issues
    if any(word in text_lower for word in ['no response', 'not responding', 'no one responds']):
        issues.append("unresponsive_service")
    
    if any(word in text_lower for word in ['rude', 'unhelpful', 'poor service']):
        issues.append("poor_service")
    
    # Shipping issues
    if any(word in text_lower for word in ['late', 'delayed', 'slow delivery']):
        issues.append("delayed_delivery")
    
    if 'tracking' in text_lower and any(word in text_lower for word in ['not updated', 'no update']):
        issues.append("tracking_not_updated")
    
    # Resolution requests
    if 'refund' in text_lower:
        issues.append("requesting_refund")
    
    if 'replacement' in text_lower:
        issues.append("requesting_replacement")
    
    return issues

def create_natural_summary(text, key_info, category, sentiment):
    """Create a natural, paragraph-style summary like a human support agent would write."""
    
    # Extract specific details
    products = detect_product_mentions(text)
    issues = extract_specific_issues(text)
    urgency = detect_urgency(text)
    text_lower = text.lower()
    
    # Build summary components
    summary_parts = []
    
    # Start with the main action/event
    if products:
        product_name = products[0]
        if 'order_numbers' in key_info:
            summary_parts.append(f"A customer placed a {product_name} order ({key_info['order_numbers'][0]})")
        else:
            summary_parts.append(f"A customer ordered a {product_name}")
    else:
        if 'order_numbers' in key_info:
            summary_parts.append(f"A customer placed an order ({key_info['order_numbers'][0]})")
        else:
            summary_parts.append("A customer placed an order")
    
    # Add financial details if available
    if 'prices' in key_info:
        if len(key_info['prices']) == 1:
            summary_parts.append(f" valued at ${key_info['prices'][0]:.0f}")
        else:
            total = sum(key_info['prices'])
            summary_parts.append(f" totaling ${total:.0f}")
    
    # Add timing information
    time_context = []
    if 'time_periods' in key_info:
        for time_detail in key_info['time_periods']:
            time_value = time_detail[0]
            time_unit = time_detail[1]
            
            # Context-sensitive timing
            if any(word in text_lower for word in ['ago', 'ordered', 'placed']):
                time_context.append(f" {time_value} {time_unit} ago")
            elif any(word in text_lower for word in ['supposed to', 'expected', 'within']):
                time_context.append(f" with an expected {time_value}-{time_unit} delivery window")
            elif any(word in text_lower for word in ['tracking', 'updated', 'status']):
                time_context.append(f" with tracking information stagnant for the past {time_value} {time_unit}")
            else:
                time_context.append(f" {time_value} {time_unit}")
    
    if time_context:
        summary_parts.append(time_context[0])
    
    # Describe the main problem
    problem_description = []
    
    if category == "shipping":
        if "delayed_delivery" in issues:
            problem_description.append(" but the product remains undelivered")
        elif "tracking_not_updated" in issues:
            problem_description.append(" but tracking information has not been updated")
        else:
            problem_description.append(" but is experiencing shipping issues")
    
    elif category == "product_defect":
        if "physical_damage" in issues:
            if products and products[0] == "laptop":
                problem_description.append(" but it arrived with a broken screen")
            else:
                problem_description.append(" but it arrived damaged")
        elif "functional_defect" in issues:
            problem_description.append(" but the product is not functioning properly")
        else:
            problem_description.append(" but received a defective product")
    
    elif category == "wrong_item":
        problem_description.append(" but received an incorrect item")
    
    elif category == "billing":
        problem_description.append(" but is experiencing billing issues")
    
    elif category == "service":
        problem_description.append(" but had poor customer service experience")
    
    # Add additional timing details if available
    if len(time_context) > 1:
        problem_description.append(f", {time_context[1]}")
    
    if problem_description:
        summary_parts.append("".join(problem_description))
    
    # Add customer service attempts
    service_attempts = []
    if "unresponsive_service" in issues:
        service_attempts.append("made multiple unsuccessful attempts to reach customer support without receiving helpful assistance")
    elif any(word in text_lower for word in ['contacted', 'tried contacting', 'reached out']):
        if any(word in text_lower for word in ['multiple', 'several', 'many times']):
            service_attempts.append("has made multiple attempts to contact customer support")
        else:
            service_attempts.append("has attempted to contact customer support")
    
    if service_attempts:
        summary_parts.append(f". The customer {service_attempts[0]}")
    
    # Add impact and urgency
    impact_parts = []
    if urgency or any(word in text_lower for word in ['affecting', 'impacting', 'need']):
        if 'work' in text_lower:
            impact_parts.append("the delay is significantly impacting their work productivity")
        elif 'business' in text_lower:
            impact_parts.append("the issue is affecting their business operations")
        else:
            impact_parts.append("the situation is causing significant inconvenience")
    
    if urgency:
        impact_parts.append("creating an urgent need for resolution")
    
    if impact_parts:
        summary_parts.append(f", and {', '.join(impact_parts)}")
    
    # Add resolution request
    resolution_request = []
    if "requesting_refund" in issues:
        resolution_request.append("seeking a refund")
    elif "requesting_replacement" in issues:
        resolution_request.append("requesting a replacement")
    elif 'resolve' in text_lower or 'fix' in text_lower:
        resolution_request.append("requesting resolution")
    
    if resolution_request:
        summary_parts.append(f". The customer is {resolution_request[0]}")
    
    # Add complaint classification
    classification_parts = []
    
    # Category description
    category_descriptions = {
        "shipping": "shipping delay complaint",
        "product_defect": "product defect complaint", 
        "billing": "billing issue complaint",
        "wrong_item": "wrong item complaint",
        "service": "customer service complaint",
        "technical": "technical issue complaint",
        "pricing": "pricing concern complaint"
    }
    
    category_desc = category_descriptions.get(category, "complaint")
    
    # Sentiment description
    sentiment_descriptions = {
        "angry": "an angry customer experiencing",
        "frustrated": "a frustrated customer experiencing", 
        "polite": "a polite customer experiencing",
        "neutral": "a customer experiencing"
    }
    
    sentiment_desc = sentiment_descriptions.get(sentiment, "a customer experiencing")
    
    classification_parts.append(f". This {category_desc} reflects {sentiment_desc}")
    
    # Add specific service failures
    service_failures = []
    if category == "shipping":
        if "delayed_delivery" in issues:
            service_failures.append("delivery delays")
        if "tracking_not_updated" in issues:
            service_failures.append("tracking system failures")
    
    if "unresponsive_service" in issues:
        service_failures.append("customer service failures")
    
    if service_failures:
        classification_parts.append(f" both {' and '.join(service_failures)}")
    else:
        classification_parts.append(" service issues")
    
    # Add required actions
    required_actions = []
    if category == "shipping":
        required_actions.append("provide updated tracking information")
        required_actions.append("expedite the shipment")
    elif category == "product_defect":
        if "requesting_refund" in issues:
            required_actions.append("process the refund request")
        else:
            required_actions.append("arrange replacement or refund")
    elif category == "billing":
        required_actions.append("review billing details")
    elif category == "service":
        required_actions.append("improve customer service response")
    
    required_actions.append("restore confidence in the support process")
    
    if required_actions:
        classification_parts.append(f", requiring immediate attention to {', '.join(required_actions[:2])}")
        if len(required_actions) > 2:
            classification_parts.append(f", and {required_actions[-1]}")
    
    summary_parts.append("".join(classification_parts))
    
    # Join everything into a single paragraph - FIXED: Use proper joining
    full_summary = "".join(summary_parts)
    
    # Clean up formatting
    full_summary = re.sub(r'\s+', ' ', full_summary)
    full_summary = full_summary.strip()
    
    # Ensure proper punctuation
    if not full_summary.endswith('.'):
        full_summary += '.'
    
    return full_summary

async def get_ai_summary(text):
    """Get AI-generated summary with improved parameters for general responses."""
    
    for model_name, model_id in SUMMARIZATION_MODELS.items():
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            # Improved prompt for more general summary
            prompt_text = f"Summarize this customer complaint in a professional, general manner focusing on the main issue and required resolution: {text}"
            
            payload = {
                "inputs": prompt_text,
                "parameters": {
                    "min_length": 80,
                    "max_length": 200,
                    "do_sample": True,
                    "length_penalty": 1.0,
                    "num_beams": 4,
                    "temperature": 0.3,  # Lower temperature for more focused output
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3
                }
            }
            
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                output = response.json()
                
                if isinstance(output, list) and len(output) > 0:
                    if 'summary_text' in output[0]:
                        ai_summary = output[0]['summary_text']
                        # Clean and generalize the AI summary
                        ai_summary = clean_ai_summary(ai_summary)
                        return ai_summary
                elif isinstance(output, dict):
                    if 'summary_text' in output:
                        ai_summary = output['summary_text']
                        # Clean and generalize the AI summary
                        ai_summary = clean_ai_summary(ai_summary)
                        return ai_summary
                
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    return None

def clean_ai_summary(summary):
    """Clean and generalize AI-generated summary."""
    if not summary:
        return None
    
    # Remove specific personal information
    summary = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email removed]', summary)
    summary = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[phone number removed]', summary)
    summary = re.sub(r'\b[A-Z]{2}\d{8,}\b', '[order number]', summary)
    
    # Remove overly specific details but keep general structure
    summary = re.sub(r'\b(please help me|help me out|email me)\b', 'seeking assistance', summary, flags=re.IGNORECASE)
    
    # Clean up grammar and make more professional
    summary = summary.replace('I am', 'The customer is')
    summary = summary.replace('I have', 'The customer has')
    summary = summary.replace('I ordered', 'The customer ordered')
    summary = summary.replace('my money', 'their money')
    summary = summary.replace('my order', 'their order')
    
    # Fix common grammar issues
    summary = re.sub(r'\s+', ' ', summary)
    summary = summary.strip()
    
    # Ensure it ends with proper punctuation
    if summary and not summary.endswith('.'):
        summary += '.'
    
    return summary

def create_personalized_response(text, key_info, category, sentiment):
    """Create a personalized response that matches the specific complaint details."""
    
    # Extract specific details from the complaint
    text_lower = text.lower()
    products = detect_product_mentions(text)
    issues = extract_specific_issues(text)
    urgency = detect_urgency(text)
    
    # Start with appropriate greeting based on sentiment
    if sentiment == "angry":
        greeting = "We sincerely apologize for your frustrating experience and completely understand your concern."
    elif sentiment == "frustrated":
        greeting = "We're truly sorry to hear about this disappointing experience."
    elif sentiment == "polite":
        greeting = "Thank you for bringing this to our attention in such a courteous manner."
    else:
        greeting = "Thank you for reaching out to us regarding your concern."
    
    # Build response parts
    response_parts = [greeting]
    
    # Address specific issues mentioned
    if "physical_damage" in issues:
        response_parts.append("We understand how disappointing it is to receive a damaged product.")
    
    if "functional_defect" in issues:
        response_parts.append("We take product functionality seriously and will address this defect immediately.")
    
    if "unresponsive_service" in issues:
        response_parts.append("We apologize that our customer service team hasn't been responsive to your previous attempts to contact us.")
    
    # Address time-specific concerns
    if 'time_periods' in key_info:
        time_detail = key_info['time_periods'][0]
        time_value = time_detail[0]
        time_unit = time_detail[1]
        
        response_parts.append(f"We understand that {time_value} {time_unit} is far too long to wait for a resolution.")
    
    # Address product-specific concerns
    if products:
        product_name = products[0]
        if category == "product_defect":
            response_parts.append(f"We're committed to ensuring your {product_name} meets our quality standards.")
        elif category == "shipping":
            response_parts.append(f"We know how important it is to receive your {product_name} promptly.")
    
    # Provide specific action based on category and details
    action_responses = {
        "product_defect": {
            "default": "We'll arrange for a full refund or replacement immediately, whichever you prefer.",
            "physical_damage": "We'll send you a replacement product right away and provide a prepaid return label for the damaged item.",
            "functional_defect": "We'll process a full refund or send a replacement unit immediately, and we'll investigate this quality issue."
        },
        "shipping": {
            "default": "We're immediately investigating your shipment status and will contact our shipping partner to get updated tracking information.",
            "delayed_delivery": "We're escalating this delivery issue to our logistics team to expedite your shipment."
        },
        "billing": {
            "default": "We're reviewing your account and billing details to resolve this matter promptly.",
        },
        "wrong_item": {
            "default": "We'll send you the correct item right away and arrange for pickup of the incorrect product at no cost to you.",
        },
        "service": {
            "default": "We're addressing this service issue with our team leadership and will ensure you receive the support you deserve.",
            "unresponsive_service": "We're escalating your case to our management team to ensure you receive immediate attention and resolution."
        },
        "technical": {
            "default": "Our technical team is prioritizing this issue and working to resolve it as quickly as possible.",
        },
        "general": {
            "default": "We're looking into your concern comprehensively and will provide a complete resolution."
        }
    }
    
    # Get specific action response
    category_actions = action_responses.get(category, {"default": "We're looking into your concern and will resolve it promptly."})
    
    # Choose most specific action available
    action_response = category_actions.get("default")
    for issue in issues:
        if issue in category_actions:
            action_response = category_actions[issue]
            break
    
    response_parts.append(action_response)
    
    # Add timeline commitment
    if urgency or sentiment == "angry":
        response_parts.append("Given the urgency of this matter, we'll provide you with an update within 4 hours.")
    elif category in ["product_defect", "wrong_item"]:
        response_parts.append("We'll process your replacement/refund within 24 hours and keep you updated throughout.")
    elif category == "shipping":
        response_parts.append("We'll update you with tracking information and expected delivery within 24 hours.")
    else:
        response_parts.append("We'll follow up with you within 24-48 hours with a complete resolution.")
    
    # Add order reference if available
    if 'order_numbers' in key_info:
        response_parts.append(f"We've noted your reference number {key_info['order_numbers'][0]} for priority handling.")
    
    # Add closing based on sentiment
    if sentiment == "angry":
        closing = "We truly value your business and are committed to making this right. Please don't hesitate to contact us if you need any immediate assistance."
    elif sentiment == "polite":
        closing = "We appreciate your patience and understanding. Thank you for giving us the opportunity to resolve this for you."
    else:
        closing = "We appreciate your business and look forward to resolving this matter to your complete satisfaction."
    
    response_parts.append(closing)
    
    # Join all parts
    full_response = " ".join(response_parts)
    
    return full_response

@app.post("/summarize")
async def summarize(complaint: ComplaintRequest):
    original_text = clean_text(complaint.text)
    original_word_count = len(original_text.split())

    if original_word_count < 3:
        return {
            "success": False,
            "error": "Text too short to summarize"
        }
    
    key_info = extract_key_info(original_text)
    category = categorize_complaint(original_text)
    sentiment = detect_sentiment(original_text)
    
    # Create natural paragraph-style summary
    summary = create_natural_summary(original_text, key_info, category, sentiment)
    
    # Try to get AI summary as additional analysis
    ai_summary = await get_ai_summary(original_text)
    
    # Clean and format final summary
    summary = clean_text(summary)
    summary_word_count = len(summary.split())

    response_data = {
        "success": True,
        "original_text": original_text,
        "original_word_count": original_word_count,
        "summary": summary,
        "summary_word_count": summary_word_count,
        "sentiment": sentiment,
        "category": category,
        "key_info": key_info
    }
    
    # Add AI summary if available
    if ai_summary:
        response_data["ai_analysis"] = ai_summary
    
    return response_data

@app.post("/respond")
async def respond(complaint: ComplaintRequest):
    cleaned = clean_text(complaint.text)
    
    key_info = extract_key_info(cleaned)
    category = categorize_complaint(cleaned)
    sentiment = detect_sentiment(cleaned)
    
    # Create personalized response
    response_text = create_personalized_response(cleaned, key_info, category, sentiment)
    
    return {
        "success": True,
        "response": response_text,
        "sentiment": sentiment,
        "category": category,
        "key_info": key_info
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "complaint-summarizer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)