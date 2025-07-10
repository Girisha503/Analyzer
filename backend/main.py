from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Optional
import nltk
import asyncio
import time
from urllib.error import URLError # Import URLError for more specific network errors

# Download necessary NLTK data (run this once)
# This block attempts to download 'averaged_perceptron_tagger' if not found.
# It uses a general Exception catch for newer NLTK versions where DownloadError might not exist.
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except Exception as e: # Catching general Exception for robust handling across NLTK versions
    print(f"NLTK 'averaged_perceptron_tagger' not found locally. Attempting to download...")
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("NLTK 'averaged_perceptron_tagger' downloaded successfully.")
    except URLError as url_err:
        print(f"Network error during NLTK download: {url_err}")
        print("Please check your internet connection. NLTK tagging might not work.")
    except Exception as download_err:
        print(f"An unexpected error occurred during NLTK download: {download_err}")
        print("Please ensure you have an active internet connection or try downloading manually using 'python -m nltk.downloader averaged_perceptron_tagger'")
    # Note: If NLTK download fails, subsequent text analysis might be less effective or error out.

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComplaintRequest(BaseModel):
    text: str

async def is_valid_complaint(text: str, headers: dict) -> bool:
    """
    Uses keyword matching and a refined two-step zero-shot classification
    to robustly identify legitimate product/service complaints in natural human language.
    """
    text_lower = text.lower()

    # --- EXPANDED: Keyword-based Pre-check for strong complaint indicators ---
    strong_complaint_keywords = [
        "shattered", "broken", "damaged", "wrong", "defect", "problem", "issue",
        "unacceptable", "faulty", "missing", "delay", "refund", "return",
        "not working", "malfunction", "poor quality", "incorrect", "defective",
        # Added more natural language indicators
        "disappointed", "frustrated", "terrible", "awful", "horrible", "bad",
        "doesn't work", "won't work", "stopped working", "never worked",
        "completely useless", "waste of money", "rip off", "scam",
        "fake", "cheap", "flimsy", "fell apart", "broke", "cracked",
        "scratched", "dented", "torn", "stained", "dirty", "smells",
        "uncomfortable", "too small", "too big", "wrong size", "wrong color",
        "late", "delayed", "never arrived", "still waiting", "where is",
        "charged twice", "overcharged", "billing error", "payment issue",
        "customer service", "rude", "unhelpful", "ignored", "no response",
        
        # ADDED: Shipping and delivery issues
        "stuck", "held up", "customs", "not delivered", "not arrived",
        "shipping problem", "delivery issue", "package stuck", "package lost",
        "tracking shows", "hasn't moved", "no update", "no progress",
        
        # Poor grammar/spelling variations
        "dont work", "doesnt work", "wont work", "cant use", "not gud", "not good",
        "very bad", "so bad", "realy bad", "really bad", "totaly broken", "totally broken",
        "completly broken", "completely broken", "stil waiting", "still waiting",
        "stil not", "still not", "havnt received", "havent received", "haven't received",
        "didnt get", "didn't get", "never got", "never recieved", "never received",
        "orderd", "ordered", "recived", "received", "deliverd", "delivered",
        "shiped", "shipped", "arived", "arrived", "broke", "broking", "broked",
        "damged", "damagd", "defectiv", "defective", "falty", "faulty",
        "problm", "problme", "isue", "isuue", "refund pls", "refund please",
        "pls help", "please help", "pls fix", "please fix", "need help",
        "very disapointed", "very disappointed", "not hapy", "not happy",
        "vry bad", "very bad", "so disapointed", "so disappointed"
    ]
    
    # Check for complaint phrases (more natural language with grammar variations)
    complaint_phrases = [
        "i'm not happy", "i am not happy", "im not happy", "not satisfied", "very disappointed",
        "this is ridiculous", "this is unacceptable", "i want my money back", "want my money back",
        "i need a refund", "need a refund", "please help", "pls help", "what can you do", 
        "resolve this", "fix this", "replace this", "exchange this", "send me a new",
        "i ordered", "i bought", "i purchased", "i received", "i got", "delivered",
        "came with", "supposed to", "expected", "should have", "promised",
        
        # ADDED: Help-seeking phrases that indicate problems
        "not sure what to do", "don't know what to do", "what should i do",
        "can you help", "need assistance", "need guidance", "stuck at",
        "package is stuck", "shipment stuck", "delivery stuck", "held at",
        "customs clearance", "customs delay", "customs issue", "customs problem",
        "urgent delivery", "needs to arrive", "time sensitive", "deadline",
        
        # Grammar variations and typos
        "i orderd", "i order", "i bough", "i buy", "i purchas", "i recieved", "i recived",
        "deliverd to", "came wit", "suposed to", "expectd", "shoud have", "promissed",
        "stil havnt", "still havent", "stil waiting", "still waitng", "wheres my",
        "where is my", "why havnt", "why havent", "when wil", "when will",
        "how can i", "what shoud i", "what should i", "pls tel me", "please tell me",
        "need to no", "need to know", "want to no", "want to know",
        "order number", "order no", "order id", "ref number", "ref no", "reference no",
        "my order", "the order", "this order", "that order", "order was", "order is",
        
        # Common complaint starters with poor grammar
        "i am writing", "i m writing", "im writing", "writing to", "contacting about",
        "reaching out", "need help with", "having problem", "having issue", "got problem",
        "there is problem", "there's problem", "problem with", "issue with", "wrong with",
        "not working", "doesnt work", "dont work", "wont work", "stopped work",
        "never work", "not function", "malfunction", "broken down", "gave up"
    ]
    
    # ADDED: Check for order reference + problem context
    has_order_reference = bool(re.search(r'(?:order|ref|reference|#|id)\s*:?\s*[A-Z0-9-]{4,}', text_lower, re.IGNORECASE))
    
    # ADDED: Check for urgency/time-sensitive context
    urgency_indicators = [
        "before next week", "by next week", "urgent", "asap", "immediately", "soon",
        "time sensitive", "deadline", "needs to arrive", "has to arrive", "must arrive",
        "gift", "birthday", "anniversary", "wedding", "event", "party"
    ]
    has_urgency = any(indicator in text_lower for indicator in urgency_indicators)
    
    # ADDED: Check for shipping/delivery context
    shipping_context = [
        "package", "shipment", "delivery", "shipping", "customs", "tracking",
        "courier", "fedex", "ups", "dhl", "usps", "post office", "mail"
    ]
    has_shipping_context = any(context in text_lower for context in shipping_context)
    
    if (any(keyword in text_lower for keyword in strong_complaint_keywords) or 
        any(phrase in text_lower for phrase in complaint_phrases) or
        # ADDED: Accept if it has order reference + shipping context + urgency
        (has_order_reference and has_shipping_context and has_urgency)):
        print("[Validation Pre-check] Strong complaint indicators detected. Bypassing AI validation.")
        return True

    try:
        classification_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

        # --- Step 1: UPDATED - More inclusive categorization ---
        step1_labels = [
            "expressing dissatisfaction or problems with a product or service",
            "asking for help or resolution with an issue",
            "sharing a negative experience about a purchase",
            "requesting assistance with a delivery or shipping problem",  # ADDED
            "seeking guidance about an order or package issue",  # ADDED
            "casual conversation or general statement",
            "asking a question without expressing problems",
            "completely unrelated content"
        ]
        
        start_time_step1 = time.perf_counter()
        step1_response_json = await asyncio.to_thread(
            lambda: requests.post(classification_url, headers=headers, json={"inputs": text, "parameters": {"candidate_labels": step1_labels}}).json()
        )
        end_time_step1 = time.perf_counter()
        print(f"[Validation Step 1] API call took {end_time_step1 - start_time_step1:.2f} seconds")

        if not step1_response_json or 'labels' not in step1_response_json or not step1_response_json['labels']:
            print(f"[Validation Step 1] API returned no valid JSON data or labels.")
            return False

        step1_top_label = step1_response_json['labels'][0]
        step1_top_score = step1_response_json['scores'][0]
        print(f"[Validation Step 1] Label: '{step1_top_label}', Score: {step1_top_score:.2f}")

        # More inclusive first step - accept any complaint-related category
        complaint_related_labels = [
            "expressing dissatisfaction or problems with a product or service",
            "asking for help or resolution with an issue", 
            "sharing a negative experience about a purchase",
            "requesting assistance with a delivery or shipping problem",  # ADDED
            "seeking guidance about an order or package issue"  # ADDED
        ]
        
        if step1_top_label in complaint_related_labels and step1_top_score > 0.3:  # Further lowered threshold
            pass  # Proceed to step 2
        elif step1_top_label == "casual conversation or general statement" and step1_top_score < 0.75:
            print(f"[Validation Step 1] Might be complaint disguised as casual conversation. Proceeding to Step 2.")
            pass
        else:
            print(f"[Validation Step 1] Clearly not a complaint. Top label: '{step1_top_label}' with score: {step1_top_score:.2f}")
            return False

        # --- Step 2: UPDATED - More natural language intent detection ---
        step2_labels = [
            "wanting something fixed, replaced, refunded, or resolved",
            "expressing strong dissatisfaction that needs attention",
            "describing a problem or issue with a product or service",
            "seeking help or guidance with a delivery or shipping issue",  # ADDED
            "requesting assistance with an urgent order or package problem",  # ADDED
            "just sharing information without wanting action",
            "asking a general question without expressing problems"
        ]
        
        start_time_step2 = time.perf_counter()
        step2_response_json = await asyncio.to_thread(
            lambda: requests.post(classification_url, headers=headers, json={"inputs": text, "parameters": {"candidate_labels": step2_labels}}).json()
        )
        end_time_step2 = time.perf_counter()
        print(f"[Validation Step 2] API call took {end_time_step2 - start_time_step2:.2f} seconds")

        if not step2_response_json or 'labels' not in step2_response_json or not step2_response_json['labels']:
            print(f"[Validation Step 2] API returned no valid JSON data or labels.")
            return False

        step2_top_label = step2_response_json['labels'][0]
        step2_top_score = step2_response_json['scores'][0]
        print(f"[Validation Step 2] Label: '{step2_top_label}', Score: {step2_top_score:.2f}")

        # Accept multiple types of complaint intent with lower thresholds
        valid_complaint_intents = [
            "wanting something fixed, replaced, refunded, or resolved",
            "expressing strong dissatisfaction that needs attention",
            "describing a problem or issue with a product or service",
            "seeking help or guidance with a delivery or shipping issue",  # ADDED
            "requesting assistance with an urgent order or package problem"  # ADDED
        ]
        
        if step2_top_label in valid_complaint_intents and step2_top_score > 0.3:  # Further lowered threshold
            print("[Validation Success] Text identified as a valid complaint based on intent.")
            return True
        
        print(f"[Validation Step 2] Intent not complaint-related. Top label: '{step2_top_label}' with score: {step2_top_score:.2f}")
        return False

    except requests.exceptions.RequestException as req_e:
        print(f"Network or API error during complaint validation: {req_e}")
        # If API fails, be more permissive with keyword-based validation
        basic_complaint_words = [
            "problem", "issue", "wrong", "broken", "defect", "bad", "terrible", 
            "disappointed", "refund", "return", "help", "fix", "replace",
            "stuck", "customs", "delayed", "package", "delivery", "shipping",  # ADDED
            # Include common typos and grammar issues
            "problm", "isue", "rong", "broking", "defectiv", "dissapointed",
            "dont work", "doesnt work", "not gud", "very bad", "so bad",
            "orderd", "recived", "deliverd", "havnt received", "stil waiting",
            "order number", "order no", "ref no", "my order", "the order"
        ]
        
        # Also check for order-related complaints even with poor grammar
        order_complaint_patterns = [
            r'(?:order|ref|oder|ordor)\s*(?:number|no|id|#)?\s*[A-Z0-9-]{4,}',
            r'(?:my|the)\s+order\s+(?:was|is|#)',
            r'(?:placed|made)\s+(?:order|an order)',
            r'(?:ordered|bought|purchased).*(?:but|and|however)',
            r'(?:received|got|delivered).*(?:wrong|broken|damaged|bad)',
            r'(?:package|shipment).*(?:stuck|delayed|lost|problem)',  # ADDED
            r'customs.*(?:stuck|held|delay|problem|issue)'  # ADDED
        ]
        
        has_order_complaint = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in order_complaint_patterns)
        
        if (any(word in text_lower for word in basic_complaint_words) or 
            has_order_complaint or
            (has_order_reference and has_shipping_context)):  # ADDED fallback condition
            print("[Validation Fallback] API failed, but found complaint indicators. Accepting as complaint.")
            return True
        return False
    except json.JSONDecodeError:
        print("Failed to decode JSON response from Hugging Face API during validation.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during complaint validation: {e}")
        return False
def extract_basic_info(text):
    """Extract structured data like prices, order numbers, etc."""
    info = {}
    
    # Extract prices
    prices = re.findall(r'\$(\d+(?:\.\d{2})?)', text)
    if prices:
        info['prices'] = [float(p) for p in prices]
    
    # Updated order number patterns with better matching
    order_patterns = [
        # Pattern 1: Order/Ref/Reference + optional (number/no/ID) + optional colon/space + alphanumeric
        r'(?:order|ref|reference|#)\s*(?:number|no\.?|id)?\s*:?\s*([A-Z0-9-]{4,})',
        # Pattern 2: Specific "Order ID" pattern (case insensitive)
        r'order\s+id\s*:?\s*([A-Z0-9-]{4,})',
        # Pattern 3: Generic alphanumeric patterns (existing patterns)
        r'\b([A-Z]{2,3}\d{6,})\b',
        r'\b(\d{8,})\b',
        # Pattern 4: Shorter numeric patterns (but more specific context)
        r'(?:order|ref|reference|#|id)\s*:?\s*(\d{4,})',
    ]
    
    # Try each pattern until we find matches
    for pattern in order_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info['order_numbers'] = matches
            break
    
    # Extract dates
    date_patterns = [
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4})\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info['dates'] = matches
            break
    
    return info

def _call_hf_api_sync(url: str, headers: dict, payload: dict) -> Optional[dict]:
    """Helper to make a synchronous request to Hugging Face API and handle common errors."""
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30) # Added timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        print(f"API call to {url} timed out after 30 seconds.")
        return None
    except requests.exceptions.RequestException as req_e:
        print(f"API call to {url} failed: {req_e}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from {url}.")
        return None
    except Exception as e:
        print(f"An unexpected error during API call to {url}: {e}")
        return None

async def extract_product_with_ai(text):
    """Use multiple AI models to extract product information accurately, with refined fallback."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Early check for generic complaints without specific product mentions
    text_lower = text.lower()
    generic_indicators = [
        "your product", "the product", "this product", "that product",
        "your service", "the service", "this service", "that service",
        "your company", "the company", "this company"
    ]
    
    # If text is very generic and short, return generic product
    if (len(text.split()) < 10 and 
        any(indicator in text_lower for indicator in generic_indicators) and
        not any(specific in text_lower for specific in ["bag", "wallet", "phone", "laptop", "chair", "vase", "jewelry", "clothing"])):
        print("[Product Extraction] Generic complaint detected, returning 'product'")
        return "product"
    
    # Method 1: Use Question-Answering model (most effective for direct product queries)
    qa_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    qa_payload = {
        "inputs": {
            "question": "What specific product, item, or object is the complaint about?",
            "context": text
        }
    }
    start_time_qa = time.perf_counter()
    qa_result = await asyncio.to_thread(_call_hf_api_sync, qa_url, headers, qa_payload)
    end_time_qa = time.perf_counter()
    print(f"[Product Extraction - QA] API call took {end_time_qa - start_time_qa:.2f} seconds")
    
    if qa_result and isinstance(qa_result, dict) and 'answer' in qa_result:
        answer = qa_result['answer'].strip().lower()
        confidence = qa_result.get('score', 0)
        print(f"[Product Extraction - QA] Answer: '{answer}', Score: {confidence:.2f}")
        
        # Enhanced generic term detection
        generic_terms = [
            'it', 'this', 'that', 'item', 'product', 'thing', 'service',
            'your product', 'the product', 'this product', 'that product',
            'your service', 'the service', 'this service', 'that service'
        ]
        
        if confidence > 0.45 and len(answer) > 2 and len(answer) < 50:
            # Check if answer is too generic
            if answer not in generic_terms and not answer.startswith(('the ', 'a ', 'an ', 'your ', 'this ', 'that ')):
                cleaned_answer = re.sub(r'^(my|the|a|an|your|this|that)\s+', '', answer).strip()
                if cleaned_answer and cleaned_answer not in generic_terms:
                    return cleaned_answer
            elif confidence > 0.8:  # Only accept generic answers with very high confidence
                return "product"
    
    # Method 2: Use NER (Named Entity Recognition) for product extraction
    ner_url = "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english"
    start_time_ner = time.perf_counter()
    ner_result = await asyncio.to_thread(_call_hf_api_sync, ner_url, headers, {"inputs": text})
    end_time_ner = time.perf_counter()
    print(f"[Product Extraction - NER] API call took {end_time_ner - start_time_ner:.2f} seconds")

    if ner_result and isinstance(ner_result, list):
        print(f"[Product Extraction - NER] Result: {ner_result}")
        products = []
        for entity in ner_result:
            if entity.get('entity_group') == 'MISC' and entity.get('score', 0) > 0.7:
                word = entity.get('word', '').replace('##', '').strip().lower()
                if len(word) > 2 and word not in ['chain', 'item', 'product', 'service']:
                    products.append(word)
        if products:
            return products[0]
    
    # Method 3: Use zero-shot classification - IMPROVED with better threshold handling
    candidate_labels = [
        "bag", "wallet", "jewelry", "clothing", "electronic device", "furniture", "home decor", 
        "book", "kitchen appliance", "beauty product", "sports equipment",
        "toy", "automotive part", "office supplies", "food item",
        "health product", "pet supplies", "tools", "garden supplies", "accessory",
        "vase", "unspecified product or service"  # Added generic option
    ]
    classification_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    classification_payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": candidate_labels
        }
    }
    start_time_classification = time.perf_counter()
    classification_result = await asyncio.to_thread(_call_hf_api_sync, classification_url, headers, classification_payload)
    end_time_classification = time.perf_counter()
    print(f"[Product Extraction - Classification] API call took {end_time_classification - start_time_classification:.2f} seconds")

    if classification_result and 'labels' in classification_result and len(classification_result['labels']) > 0:
        print(f"[Product Extraction - Classification] Result: {classification_result}")
        top_category = classification_result['labels'][0]
        confidence = classification_result['scores'][0]
        
        # IMPROVED: Higher threshold for generic complaints and better handling
        if confidence > 0.7:  # Increased threshold
            if top_category == "unspecified product or service":
                return "product"
            else:
                specific_item = await extract_specific_item_from_category(text, top_category, headers)
                if specific_item:
                    return specific_item
                else:
                    return top_category.replace(' item', '').replace(' device', '').replace(' a ', '').replace(' an ', '')
        elif confidence > 0.5:
            # Medium confidence - check if it's the generic category
            if top_category == "unspecified product or service":
                return "product"
            # Otherwise, be more cautious and return generic
            else:
                return "product"
    
    # Method 4: Use text summarization - IMPROVED with better filtering
    summary_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    product_focused_text = f"The customer's complaint is specifically about: {text}. What is the main item or product they are referring to?"
    summary_payload = {
        "inputs": product_focused_text,
        "parameters": {
            "min_length": 5,
            "max_length": 20,
            "do_sample": True,
            "temperature": 0.3
        }
    }
    start_time_summary = time.perf_counter()
    summary_result = await asyncio.to_thread(_call_hf_api_sync, summary_url, headers, summary_payload)
    end_time_summary = time.perf_counter()
    print(f"[Product Extraction - Summary] API call took {end_time_summary - start_time_summary:.2f} seconds")

    if summary_result and isinstance(summary_result, list) and len(summary_result) > 0:
        print(f"[Product Extraction - Summary] Result: {summary_result}")
        summary = summary_result[0].get('summary_text', '').lower()
        cleaned_summary = re.sub(r'^(the|a|an|your|this|that)\s+', '', summary).strip()
        
        # Better filtering for generic responses
        generic_summary_terms = ['item', 'product', 'service', 'complaint', 'issue', 'customer', 'main']
        if cleaned_summary and cleaned_summary not in generic_summary_terms:
            return cleaned_summary
    
    # Final Fallback: Use NLTK for better noun extraction - IMPROVED
    nltk_result = extract_nouns_from_text(text)
    if nltk_result != "item":  # Only use if NLTK found something specific
        return nltk_result
    
    # Ultimate fallback for truly generic complaints
    return "product"

async def extract_specific_item_from_category(text, category, headers):
    """Extract specific item within a product category using QA."""
    qa_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    qa_payload = {
        "inputs": {
            "question": f"What specific item (e.g., 'sling bag', 'iphone', 'chair', 'ceramic vase') related to a '{category}' is mentioned?",
            "context": text
        }
    }
    start_time_specific_qa = time.perf_counter()
    qa_result = await asyncio.to_thread(_call_hf_api_sync, qa_url, headers, qa_payload)
    end_time_specific_qa = time.perf_counter()
    print(f"[Specific Item from Category QA] API call took {end_time_specific_qa - start_time_specific_qa:.2f} seconds")

    if qa_result and isinstance(qa_result, dict) and 'answer' in qa_result:
        answer = qa_result['answer'].strip().lower()
        confidence = qa_result.get('score', 0)
        print(f"[Specific Item from Category QA] Answer: '{answer}', Score: {confidence:.2f}")
        generic_terms = ['it', 'this', 'that', 'item', 'product', 'thing', 'bag', 'service']
        if confidence > 0.35 and len(answer) > 2 and len(answer) < 30 and answer not in generic_terms:
            cleaned_answer = re.sub(r'^(my|the|a|an)\s+', '', answer).strip()
            if cleaned_answer:
                return cleaned_answer
    
    return None

def extract_nouns_from_text(text):
    """Extract potential product nouns from text using NLTK as final fallback."""
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'were', 'are', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'order', 'ordered', 'need', 'free', 'correct', 'immediately', 'not', 'was', 'color', 'wrong', 'issue', 'problem', 'customer', 'service', 'received', 'delivered', 'yesterday', 'today', 'bag',
        'product', 'item', 'thing', 'chain', # Add 'chain' here to ensure it's not picked up as the *main product* if it's just a component
        'my', 'your', 'his', 'her', 'its'
    }
    
    potential_products = []
    for word in nouns:
        if word not in stop_words and len(word) > 2:
            potential_products.append(word)
    
    if potential_products:
        return potential_products[0]
    
    return "item"

async def analyze_customer_sentiment(text, headers):
    """
    Enhanced sentiment analysis with multiple approaches and contextual understanding
    """
    text_lower = text.lower()
    
    # Enhanced keyword-based sentiment indicators with weights
    sentiment_indicators = {
        'extremely_negative': {
            'keywords': [
                'furious', 'outraged', 'disgusted', 'livid', 'enraged', 'incensed',
                'absolutely horrible', 'completely unacceptable', 'utterly disappointed',
                'totally disgusted', 'extremely frustrated', 'beyond angry',
                'worst experience ever', 'never again', 'absolutely terrible',
                'completely worthless', 'total waste', 'absolute garbage',
                'demanding refund', 'want my money back now', 'this is ridiculous',
                'completely fed up', 'had enough', 'last straw'
            ],
            'patterns': [
                r'absolutely\s+(terrible|awful|horrible|disgusting)',
                r'completely\s+(unacceptable|worthless|useless)',
                r'totally\s+(disappointed|frustrated|angry)',
                r'never\s+(again|buying|ordering)',
                r'worst\s+(experience|service|product)',
                r'(hate|despise|loathe)\s+(this|your)',
                r'(demanding|want|need)\s+(refund|money\s+back)\s+(now|immediately)'
            ],
            'weight': 1.0
        },
        'very_negative': {
            'keywords': [
                'angry', 'frustrated', 'disappointed', 'upset', 'mad', 'irritated',
                'annoyed', 'unsatisfied', 'dissatisfied', 'unhappy', 'terrible',
                'awful', 'horrible', 'bad', 'poor', 'worst', 'hate', 'disgusting',
                'unacceptable', 'ridiculous', 'pathetic', 'useless', 'worthless',
                'defective', 'broken', 'damaged', 'wrong', 'incorrect', 'faulty',
                'not working', 'doesnt work', 'wont work', 'stopped working',
                'waste of money', 'rip off', 'scam', 'fraud', 'cheap', 'flimsy'
            ],
            'patterns': [
                r'(very|really|extremely|super)\s+(bad|terrible|awful|horrible)',
                r'(not|never)\s+(happy|satisfied|pleased)',
                r'(so|very)\s+(disappointed|frustrated|angry|upset)',
                r'(totally|completely)\s+(broken|useless|worthless)',
                r'(waste\s+of|wasted)\s+(money|time)',
                r'(rip\s+off|ripoff|scam)'
            ],
            'weight': 0.9
        },
        'moderately_negative': {
            'keywords': [
                'disappointed', 'concerned', 'worried', 'troubled', 'bothered',
                'not happy', 'not pleased', 'not satisfied', 'problem', 'issue',
                'concern', 'difficulty', 'trouble', 'inconvenience', 'delay',
                'missing', 'wrong', 'incorrect', 'defect', 'fault', 'error',
                'mistake', 'not working properly', 'not as expected',
                'below expectations', 'could be better', 'needs improvement'
            ],
            'patterns': [
                r'(not|never)\s+(what|as)\s+(expected|advertised|promised)',
                r'(below|under)\s+expectations',
                r'(could|should)\s+be\s+better',
                r'(needs|requires)\s+(improvement|fixing)',
                r'(having|experiencing)\s+(problems|issues|difficulties)',
                r'(a\s+bit|somewhat|slightly)\s+(disappointed|concerned|worried)'
            ],
            'weight': 0.7
        },
        'neutral_negative': {
            'keywords': [
                'question', 'inquiry', 'wondering', 'curious', 'asking',
                'would like to know', 'need information', 'seeking clarification',
                'checking status', 'following up', 'update request',
                'minor issue', 'small problem', 'quick question'
            ],
            'patterns': [
                r'(quick|simple|small)\s+(question|issue|problem)',
                r'(just|only)\s+(wondering|asking|checking)',
                r'(would|could)\s+(like|you)\s+(to|please)',
                r'(need|seeking|requesting)\s+(information|clarification|help)',
                r'(following|checking)\s+up\s+on',
                r'(status|update)\s+(on|about|regarding)'
            ],
            'weight': 0.5
        },
        'positive': {
            'keywords': [
                'thank you', 'thanks', 'appreciate', 'grateful', 'pleased',
                'happy', 'satisfied', 'good', 'great', 'excellent', 'wonderful',
                'amazing', 'fantastic', 'love', 'impressed', 'delighted',
                'professional', 'helpful', 'courteous', 'polite', 'kind',
                'understanding', 'patient', 'please help', 'if possible',
                'when convenient', 'at your earliest convenience'
            ],
            'patterns': [
                r'thank\s+(you|u)\s+(for|very|so\s+much)',
                r'(really|very|much)\s+(appreciate|grateful)',
                r'(please|kindly|would\s+you)\s+(help|assist)',
                r'(if|when)\s+(possible|convenient)',
                r'(at\s+your|your)\s+(earliest\s+)?convenience',
                r'(hope|trust)\s+(you|this)\s+(can|will)',
                r'(looking\s+forward|hope)\s+to\s+(hearing|resolution)'
            ],
            'weight': 0.8
        }
    }
    
    # Calculate sentiment scores
    sentiment_scores = {}
    
    for sentiment, indicators in sentiment_indicators.items():
        score = 0
        matches = []
        
        # Check keywords
        for keyword in indicators['keywords']:
            if keyword in text_lower:
                score += indicators['weight']
                matches.append(keyword)
        
        # Check patterns
        for pattern in indicators['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += indicators['weight'] * 1.2  # Patterns get slight boost
                matches.append(f"pattern: {pattern}")
        
        sentiment_scores[sentiment] = {
            'score': score,
            'matches': matches,
            'weight': indicators['weight']
        }
    
    # Determine primary sentiment
    max_score = max(sentiment_scores.values(), key=lambda x: x['score'])['score']
    primary_sentiment = max(sentiment_scores.keys(), key=lambda x: sentiment_scores[x]['score'])
    
    # Calculate confidence based on score difference
    sorted_scores = sorted(sentiment_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    confidence = min(0.95, max(0.3, (sorted_scores[0][1]['score'] - sorted_scores[1][1]['score']) / 3))
    
    # Contextual adjustments
    context_adjustments = {
        'urgency_boost': ['urgent', 'immediately', 'asap', 'right now', 'emergency'],
        'politeness_boost': ['please', 'kindly', 'would appreciate', 'thank you'],
        'frustration_boost': ['multiple times', 'again and again', 'over and over', 'repeatedly'],
        'severity_boost': ['completely', 'totally', 'absolutely', 'extremely', 'utterly']
    }
    
    # Apply contextual adjustments
    if any(word in text_lower for word in context_adjustments['urgency_boost']):
        if primary_sentiment in ['moderately_negative', 'very_negative']:
            confidence += 0.1
    
    if any(word in text_lower for word in context_adjustments['politeness_boost']):
        if primary_sentiment in ['neutral_negative', 'positive']:
            confidence += 0.1
    
    if any(phrase in text_lower for phrase in context_adjustments['frustration_boost']):
        if primary_sentiment in ['very_negative', 'extremely_negative']:
            confidence += 0.15
    
    # Use AI model for validation if score is close or confidence is low
    ai_sentiment = None
    ai_confidence = 0
    
    if confidence < 0.6 or max_score < 2:
        try:
            sentiment_url = "https://api-inference.huggingface.co/models/tabularisai/robust-sentiment-analysis"
            sentiment_response = await asyncio.to_thread(
                _call_hf_api_sync, sentiment_url, headers, {"inputs": text}
            )
            
            if sentiment_response and isinstance(sentiment_response, list) and len(sentiment_response) > 0:
                sentiment_data = sentiment_response[0]
                best_sentiment = max(sentiment_data, key=lambda x: x['score'])
                
                sentiment_mapping = {
    'Very Negative': 'very_negative',      
    'Negative': 'very_negative',           
    'Neutral': 'neutral_negative',         
    'Positive': 'positive',                
    'Very Positive': 'positive'            
}
                
                ai_sentiment = sentiment_mapping.get(best_sentiment['label'], 'neutral_negative')
                ai_confidence = best_sentiment['score']
                
                # Blend rule-based and AI results
                if ai_confidence > 0.8:
                    primary_sentiment = ai_sentiment
                    confidence = ai_confidence
                elif ai_confidence > 0.6 and confidence < 0.5:
                    primary_sentiment = ai_sentiment
                    confidence = (ai_confidence + confidence) / 2
                    
        except Exception as e:
            print(f"AI sentiment analysis failed: {e}")
    
    # Map to final sentiment categories
    final_sentiment_mapping = {
        'extremely_negative': 'angry',
        'very_negative': 'frustrated',
        'moderately_negative': 'disappointed',
        'neutral_negative': 'neutral',
        'positive': 'polite'
    }
    
    final_sentiment = final_sentiment_mapping.get(primary_sentiment, 'neutral')
    
    # Determine emotion based on sentiment and context
    emotion_mapping = {
        'angry': 'extremely_frustrated',
        'frustrated': 'very_frustrated', 
        'disappointed': 'concerned',
        'neutral': 'matter_of_fact',
        'polite': 'professional'
    }
    
    emotion = emotion_mapping.get(final_sentiment, 'neutral')
    
    print(f"[Enhanced Sentiment Analysis] Primary: {primary_sentiment} -> Final: {final_sentiment}")
    print(f"[Enhanced Sentiment Analysis] Emotion: {emotion}, Confidence: {confidence:.2f}")
    print(f"[Enhanced Sentiment Analysis] Top matches: {sentiment_scores[primary_sentiment]['matches'][:3]}")
    
    return {
        'sentiment': final_sentiment,
        'emotion': emotion,
        'confidence': confidence,
        'primary_indicators': sentiment_scores[primary_sentiment]['matches'][:5],
        'ai_validation': {'sentiment': ai_sentiment, 'confidence': ai_confidence} if ai_sentiment else None
    }

def generate_enhanced_human_readable_summary(text, analysis, key_info):
    """
    Enhanced summary generation with better sentiment understanding
    """
    text_lower = text.lower()
    
    # More nuanced emotion descriptions based on enhanced sentiment analysis
    emotion_descriptions = {
        'angry': 'extremely frustrated and demanding immediate action',
        'frustrated': 'very frustrated and seeking prompt resolution',
        'disappointed': 'disappointed and concerned about the issue',
        'neutral': 'matter-of-fact in their communication',
        'polite': 'professional and courteous despite the issue'
    }
    
    # Enhanced resolution detection
    resolution_patterns = {
        'immediate_replacement': ['replacement immediately', 'replace right now', 'new one asap'],
        'urgent_replacement': ['replacement', 'replace', 'new one', 'exchange'],
        'full_refund': ['full refund', 'complete refund', 'money back', 'return money'],
        'partial_refund': ['partial refund', 'some money back', 'compensation'],
        'store_credit': ['store credit', 'credit', 'voucher'],
        'repair_fix': ['repair', 'fix', 'get it working'],
        'explanation': ['explanation', 'why', 'what happened', 'how did this happen'],
        'apology': ['apology', 'sorry', 'acknowledge'],
        'better_service': ['better service', 'improve service', 'train staff'],
        'escalation': ['manager', 'supervisor', 'escalate', 'higher up']
    }
    
    desired_resolutions = []
    for resolution_type, keywords in resolution_patterns.items():
        if any(keyword in text_lower for keyword in keywords):
            desired_resolutions.append(resolution_type)
    
    # Determine primary resolution request
    if not desired_resolutions:
        if analysis['sentiment'] in ['angry', 'frustrated']:
            primary_resolution = 'immediate_action'
        elif analysis['sentiment'] == 'disappointed':
            primary_resolution = 'satisfactory_resolution'
        else:
            primary_resolution = 'assistance'
    else:
        primary_resolution = desired_resolutions[0]
    
    # Resolution text mapping
    resolution_text = {
        'immediate_replacement': 'an immediate replacement',
        'urgent_replacement': 'a replacement as soon as possible',
        'full_refund': 'a full refund',
        'partial_refund': 'compensation or partial refund',
        'store_credit': 'store credit or exchange',
        'repair_fix': 'the item to be repaired or fixed',
        'explanation': 'an explanation of what went wrong',
        'apology': 'an acknowledgment and apology',
        'better_service': 'improved customer service',
        'escalation': 'escalation to management',
        'immediate_action': 'immediate action to resolve the issue',
        'satisfactory_resolution': 'a satisfactory resolution',
        'assistance': 'assistance with their concern'
    }
    
    # Enhanced urgency detection
    urgency_phrases = {
        'critical': ['emergency', 'urgent', 'immediately', 'right now', 'asap', 'critical'],
        'high': ['soon', 'quickly', 'prompt', 'fast', 'priority', 'time sensitive'],
        'medium': ['reasonable time', 'timely', 'when possible'],
        'low': ['eventually', 'when convenient', 'no rush']
    }
    
    detected_urgency = 'standard'
    for urgency_level, phrases in urgency_phrases.items():
        if any(phrase in text_lower for phrase in phrases):
            detected_urgency = urgency_level
            break
    
    # Build enhanced summary
    product_name = analysis["product"]
    emotion_desc = emotion_descriptions.get(analysis['sentiment'], 'neutral in their communication')
    resolution_desc = resolution_text.get(primary_resolution, 'a resolution to their issue')
    
    # Issue description with more context
    issue_descriptions = {
        'wrong_item': f'they received the wrong {product_name}',
        'product_defect': f'their {product_name} has a quality defect or damage',
        'missing_parts': f'their {product_name} is missing components',
        'shipping_delay': f'their {product_name} delivery has been delayed',
        'billing_issue': f'there is a billing problem with their {product_name} purchase',
        'service_complaint': f'they experienced poor customer service regarding their {product_name}',
        'technical_issue': f'their {product_name} is malfunctioning',
        'return_request': f'they want to return their {product_name}',
        'general_inquiry': f'they have a question about their {product_name}'
    }
    
    # Add specific details based on text analysis
    specific_details = ""
    if analysis['category'] == 'product_defect':
        defect_keywords = {
            'not functioning': ['not working', 'stopped working', 'malfunctioning', 'broken'],
            'physical damage': ['cracked', 'broken', 'shattered', 'damaged', 'dented'],
            'quality issues': ['cheap', 'flimsy', 'poor quality', 'defective'],
            'component failure': ['chain not working', 'button broken', 'zipper stuck']
        }
        
        for defect_type, keywords in defect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                specific_details = f" ({defect_type})"
                break
    
    # Contact history detection
    contact_history = ""
    if 'contacted' in text_lower or 'called' in text_lower or 'emailed' in text_lower:
        if any(word in text_lower for word in ['multiple', 'several', 'many', 'numerous']):
            contact_history = " They have contacted support multiple times without resolution."
        elif any(word in text_lower for word in ['twice', 'second time', 'again']):
            contact_history = " This is their second attempt to resolve this issue."
        else:
            contact_history = " They have previously contacted support about this matter."
    
    # Build final summary
    issue_desc = issue_descriptions.get(analysis['category'], f'they have an issue with their {product_name}') + specific_details
    
    urgency_modifier = ""
    if detected_urgency == 'critical':
        urgency_modifier = "urgent "
    elif detected_urgency == 'high':
        urgency_modifier = "prompt "
    
    summary = (
        f"The customer reports that {issue_desc}. "
        f"They are {emotion_desc}.{contact_history} "
        f"They are requesting {urgency_modifier}{resolution_desc}"
    )
    
    # Add order reference if available
    if key_info.get('order_numbers'):
        summary += f" for order {key_info['order_numbers'][0]}"
    
    summary += "."
    
    return summary

# Update the main analyze_with_ai_classification function to use enhanced sentiment
async def analyze_with_ai_classification(text):
    """Enhanced AI classification with improved sentiment analysis."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    analysis_results = {
        'sentiment': 'neutral',
        'sentiment_confidence': 0.5,
        'category': 'general',
        'category_confidence': 0.5,
        'product': 'item',
        'urgency': 'standard',
        'emotion': 'neutral'
    }

    # Use enhanced sentiment analysis
    sentiment_result = await analyze_customer_sentiment(text, headers)
    analysis_results['sentiment'] = sentiment_result['sentiment']
    analysis_results['sentiment_confidence'] = sentiment_result['confidence']
    analysis_results['emotion'] = sentiment_result['emotion']
    
    # Rest of the function remains the same for category and product extraction
    category_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    category_labels = [
        "wrong item received or incorrect product",
        "product quality defect or damage",
        "missing parts or components",
        "shipping delay or delivery issue", 
        "billing or payment problem",
        "customer service complaint",
        "technical website issue",
        "return or refund request",
        "general inquiry or question"
    ]

    # Create tasks for concurrent execution
    product_task = extract_product_with_ai(text)
    category_task = asyncio.to_thread(_call_hf_api_sync, category_url, headers, 
                                     {"inputs": text, "parameters": {"candidate_labels": category_labels}})

    start_time_concurrent = time.perf_counter()
    product_result, category_response_json = await asyncio.gather(product_task, category_task)
    end_time_concurrent = time.perf_counter()
    print(f"[Concurrent AI Analysis] Product and category extraction took {end_time_concurrent - start_time_concurrent:.2f} seconds")

    # Process results
    analysis_results['product'] = product_result
    
    if category_response_json and 'labels' in category_response_json:
        top_category = category_response_json['labels'][0]
        category_mapping = {
            "wrong item received or incorrect product": "wrong_item",
            "product quality defect or damage": "product_defect",
            "missing parts or components": "missing_parts",
            "shipping delay or delivery issue": "shipping_delay",
            "billing or payment problem": "billing_issue",
            "customer service complaint": "service_complaint",
            "technical website issue": "technical_issue",
            "return or refund request": "return_request",
            "general inquiry or question": "general_inquiry"
        }
        analysis_results['category'] = category_mapping.get(top_category, 'general_inquiry')
        analysis_results['category_confidence'] = category_response_json['scores'][0]

    # Enhanced urgency detection
    urgency_indicators = {
        'critical': ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'right now'],
        'high': ['soon', 'quickly', 'prompt', 'fast', 'expedite', 'priority'],
        'medium': ['timely', 'reasonable time', 'when possible'],
        'standard': []
    }
    
    text_lower = text.lower()
    for urgency, keywords in urgency_indicators.items():
        if any(keyword in text_lower for keyword in keywords):
            analysis_results['urgency'] = urgency
            break
    
    return analysis_results

async def generate_email_response(text, analysis, key_info):
    """Generate a professional response (not email format)"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    greeting_templates = {
    'angry': 'We sincerely apologize for this frustrating experience and understand your urgency regarding your {product}.',
    'disappointed': 'We understand your disappointment regarding your {product} and want to resolve this matter promptly.',
    # 'mildly_negative': 'Thank you for bringing the issue with your {product} to our attention.', # You can remove or comment this line
    'polite': 'Thank you for your patience and for contacting us professionally regarding your {product}.',
    'neutral': 'Thank you for reaching out to us regarding your recent experience with your {product}.'
}
    
    action_templates = {
        'wrong_item': f'We will immediately send you the correct {analysis["product"]} via express shipping at no additional cost.',
        'product_defect': f'We will arrange a replacement {analysis["product"]} and provide a prepaid return label for the defective item.',
        'missing_parts': f'We will promptly ship the missing parts for your {analysis["product"]}.',
        'shipping_delay': f'We are prioritizing your {analysis["product"]} shipment and will provide tracking updates as soon as possible.',
        'billing_issue': f'Our billing team is reviewing your {analysis["product"]} charges and will resolve any discrepancies within [expected timeframe].',
        'service_complaint': f'Our customer service manager will personally address your {analysis["product"]} experience and contact you shortly.',
        'technical_issue': f'Our technical support team will contact you to resolve your {analysis["product"]} issues and ensure it functions correctly.',
        'return_request': f'We will process your {analysis["product"]} return and provide detailed instructions for sending it back.',
        'general_inquiry': f'We will provide you with detailed information about your {analysis["product"]} and address your question fully.'
    }
    
    urgency_timelines = {
        'critical': 'within 1 hour',
        'high': 'within 2 business hours',
        'medium': 'within 4 business hours',
        'standard': 'within 24 business hours'
    }
    
    product_for_response = analysis['product'] if analysis['product'] != 'item' else 'product'
    
    greeting = greeting_templates.get(analysis['sentiment'], 'Thank you for contacting us.').format(product=product_for_response)
    action = action_templates.get(analysis['category'], f'We will look into your {product_for_response} concern and get back to you.').format(product=product_for_response)
    timeline = urgency_timelines.get(analysis['urgency'], 'promptly')
    
    order_ref = ""
    if key_info.get('order_numbers'):
        order_ref = f"\n\nOrder Reference: {key_info['order_numbers'][0]}."
    
    response_body = f"""{greeting}

{action} We will follow up with you {timeline} with a complete resolution and any necessary updates.

Our team is committed to ensuring your complete satisfaction, and we appreciate your business.{order_ref}

If you have any immediate questions, please don't hesitate to contact our customer service team.

Best regards,
Customer Service Team"""
    
    return response_body

@app.post("/summarize")
async def summarize(complaint: ComplaintRequest):
    text = complaint.text.strip()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    start_total_time = time.perf_counter() # Start overall timing

    if len(text.split()) < 3:
        print("Input text too short.")
        return {
            "success": True,
            "summary": "The provided text is too short to analyze. Please provide more details about the issue.",
            "ai_summary": "Text too short.",
            "sentiment": "irrelevant",
            "category": "irrelevant",
            "product": "n/a",
            "urgency": "n/a",
            "key_info": {}
        }
    
    # Gracefully handle non-complaint text using the improved function
    start_validation_time = time.perf_counter()
    is_complaint = await is_valid_complaint(text, headers)
    end_validation_time = time.perf_counter()
    print(f"Validation step took {end_validation_time - start_validation_time:.2f} seconds.")

    if not is_complaint:
        print("Input is not a valid complaint according to validation steps.")
        return {
            "success": True,
            "summary": "This AI is specialized in analyzing customer complaints. The provided text does not appear to be related to a product or service issue or is not seeking resolution.",
            "ai_summary": "Input does not appear to be a customer complaint.",
            "sentiment": "irrelevant",
            "category": "irrelevant",
            "product": "n/a",
            "urgency": "n/a",
            "key_info": {}
        }

    try:
        key_info = extract_basic_info(text)
        
        start_analysis_time = time.perf_counter()
        analysis = await analyze_with_ai_classification(text)
        end_analysis_time = time.perf_counter()
        print(f"Analysis step took {end_analysis_time - start_analysis_time:.2f} seconds.")

        ai_summary = generate_enhanced_human_readable_summary(text, analysis, key_info)
        
        category_display = {
            'wrong_item': 'Wrong Item Received',
            'product_defect': 'Product Quality Issue',
            'missing_parts': 'Missing Parts/Components',
            'shipping_delay': 'Shipping/Delivery Issue',
            'billing_issue': 'Billing Problem',
            'service_complaint': 'Customer Service Issue',
            'technical_issue': 'Technical Problem',
            'return_request': 'Return/Refund Request',
            'general_inquiry': 'General Inquiry'
        }
        
        sentiment_display = {
            'angry': 'Highly Frustrated',
            'disappointed': 'Disappointed',
            'mildly_negative': 'Mildly Dissatisfied',
            'polite': 'Professional/Polite',
            'neutral': 'Neutral',
            'irrelevant': 'Irrelevant'
        }
        
        urgency_display = {
            'critical': 'Critical - Immediate Action Required',
            'high': 'High Priority',
            'medium': 'Medium Priority',
            'standard': 'Standard Priority',
            'n/a': 'N/A'
        }
        
        extracted_info = []
        if key_info.get('order_numbers'):
            extracted_info.append(f"Order Number: {key_info['order_numbers'][0]}")
        if key_info.get('prices'):
            extracted_info.append(f"Price Mentioned: ${key_info['prices'][0]}")
        if key_info.get('dates'):
            extracted_info.append(f"Date Referenced: {key_info['dates'][0]}")
        
        structured_summary = f"""📋 **CUSTOMER COMPLAINT ANALYSIS**

**Issue Summary:**
{ai_summary}

**📊 Analysis Details:**

• **Product/Item:** {analysis['product'].title()}

• **Issue Category:** {category_display.get(analysis['category'], 'General')}

• **Customer Sentiment:** {sentiment_display.get(analysis['sentiment'], 'Neutral')} 

• **Priority Level:** {urgency_display.get(analysis['urgency'], 'Standard')}

**📋 Extracted Information:**

{chr(10).join([f'• {info}' for info in extracted_info]) if extracted_info else '• No specific order details found'}
"""
        
        end_total_time = time.perf_counter()
        print(f"Total /summarize request time: {end_total_time - start_total_time:.2f} seconds.")

        return {
            "success": True,
            "original_text": text,
            "summary": structured_summary.strip(),
            "ai_summary": ai_summary,
            "sentiment": analysis['sentiment'],
            "category": analysis['category'],
            "product": analysis['product'],
            "urgency": analysis['urgency'],
            "key_info": key_info
        }
        
    except Exception as e:
        print(f"Error during summarization process: {e}")
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }

@app.post("/respond")
async def respond(complaint: ComplaintRequest):
    text = complaint.text.strip()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    start_total_time = time.perf_counter() # Start overall timing

    if len(text.split()) < 3:
        print("Input text too short for response generation.")
        return {
            "success": True,
            "response": "The provided text is too short to generate a meaningful response.",
            "sentiment": "irrelevant",
            "category": "irrelevant",
            "product": "n/a",
            "urgency": "n/a",
            "analysis_confidence": {"sentiment": 0.0, "category": 0.0}
        }
    
    start_validation_time = time.perf_counter()
    is_complaint = await is_valid_complaint(text, headers)
    end_validation_time = time.perf_counter()
    print(f"Validation step took {end_validation_time - start_validation_time:.2f} seconds.")

    if not is_complaint:
        print("Input is not a valid complaint for response generation.")
        return {
            "success": True,
            "response": "I'm sorry, I can only generate responses for customer complaints about products or services. The provided text doesn't seem to be a complaint or is not seeking resolution.",
            "sentiment": "irrelevant",
            "category": "irrelevant",
            "product": "n/a",
            "urgency": "n/a",
            "analysis_confidence": {"sentiment": 1.0, "category": 1.0}
        }

    try:
        key_info = extract_basic_info(text)
        
        start_analysis_time = time.perf_counter()
        analysis = await analyze_with_ai_classification(text)
        end_analysis_time = time.perf_counter()
        print(f"Analysis step took {end_analysis_time - start_analysis_time:.2f} seconds.")

        response_text = await generate_email_response(text, analysis, key_info)
        
        end_total_time = time.perf_counter()
        print(f"Total /respond request time: {end_total_time - start_total_time:.2f} seconds.")

        return {
            "success": True,
            "response": response_text,
            "sentiment": analysis['sentiment'],
            "category": analysis['category'],
            "product": analysis['product'],
            "urgency": analysis['urgency'],
            "analysis_confidence": {
                "sentiment": analysis.get('sentiment_confidence', 0),
                "category": analysis.get('category_confidence', 0)
            }
        }
        
    except Exception as e:
        print(f"Error during response generation process: {e}")
        return {
            "success": False,
            "error": f"Response generation failed: {str(e)}"
        }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.3"} # Updated version

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)