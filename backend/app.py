from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import json
from datetime import datetime
from functools import wraps
from collections import defaultdict
import time
import re
import os
import json
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import tempfile
import ffmpeg

# Language Detection
try:
    from langdetect import detect
except ImportError:
    def detect(text):
        return 'en'  # Default fallback
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import uuid
import json
import os
import sys
import threading
import time

# Configure logging - use file handler to avoid watchdog restart issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global storage for trained models
EMBEDDINGS_MODEL = None
VECTORSTORES = {}

# Global constants
SUPPORTED_BRANDS = ["samsung"]
SUPPORTED_LANGUAGES = ["en", "hi"]

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Verify API key is present
if 'GROQ_API_KEY' not in os.environ:
    logger.warning("WARNING: GROQ_API_KEY not found. Please check your .env file.")

app = Flask(__name__)
CORS(app)

# Speech Recognition Function
def speech_to_text(timeout=3, phrase_time_limit=3):
    """Perform speech recognition with configurable timeout and phrase time limit
    
    Args:
        timeout (int): Maximum time to wait for speech input
        phrase_time_limit (int): Maximum time to record a single phrase
    
    Returns:
        str: Recognized text or error message
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            # Listen with timeout and phrase time limit
            audio = recognizer.listen(
                source, 
                timeout=timeout, 
                phrase_time_limit=phrase_time_limit
            )
            
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            return text
        
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
        except sr.UnknownValueError:
            return "Could not understand audio. Please speak clearly."
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"

# Text to Speech Function
def text_to_speech(text, language='en'):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            # Generate speech
            tts = gTTS(text=text, lang=language)
            tts.save(temp_audio.name)
            
            # Play the audio
            audio = AudioSegment.from_mp3(temp_audio.name)
            play(audio)
            
            return temp_audio.name
    except Exception as e:
        print(f"Text to speech error: {e}")
        return None

### --- Complaint Registration Endpoint --- ###
@app.route('/api/complaints', methods=['POST'])
def register_complaint():
    data = request.get_json()
    complaint_id = f"CMP{uuid.uuid4().hex[:6].upper()}"
    data["complaint_id"] = complaint_id

    if not os.path.exists("complaints.json"):
        with open("complaints.json", "w") as f:
            json.dump([], f)

    with open("complaints.json", "r+") as f:
        complaints = json.load(f)
        complaints.append(data)
        f.seek(0)
        json.dump(complaints, f, indent=2)

    return jsonify({"status": "success", "complaint_id": complaint_id})


### --- FAQ RAG Chat Endpoint --- ###
# Load multilingual FAQ data
def load_faq_documents(language="en"):
    """Load and process FAQ documents from the JSON file for the specified language"""
    try:
        with open("faq_data.json", "r", encoding="utf-8") as f:
            faq_list = json.load(f)

        key_q = "question_en" if language == "en" else "question_hi"
        key_a = "answer_en" if language == "en" else "answer_hi"

        docs = [
            Document(
                page_content=f"{faq[key_q]} {faq[key_a]}",
                metadata={"source": "samsung_faq", "brand": "samsung"}
            )
            for faq in faq_list if key_q in faq and key_a in faq
        ]

        if not docs:
            logger.warning(f"Warning: No documents found for language {language}")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        return splitter.split_documents(docs)
    except Exception as e:
        logger.error(f"Error loading FAQ documents: {e}")
        return []

# Initialize the embeddings model (called only once at startup)
def initialize_embeddings():
    """Initialize the embeddings model once at startup"""
    global EMBEDDINGS_MODEL
    
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        
        logger.info("Initializing embeddings model...")
        
        # Explicitly load the model and move to CPU
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model = model.to('cpu')
        
        # Create HuggingFaceEmbeddings with the loaded model
        EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'model': model}
        )
        
        logger.info("Embeddings model initialized successfully")
        return EMBEDDINGS_MODEL
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize embeddings model: {e}", exc_info=True)
        
        # Fallback mechanism
        try:
            # Try a different approach
            from langchain_community.embeddings import HuggingFaceEmbeddings
            EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Fallback embeddings model initialized successfully")
            return EMBEDDINGS_MODEL
        except Exception as fallback_error:
            logger.error(f"ERROR: Fallback embeddings initialization failed: {fallback_error}", exc_info=True)
            return None

# Train and save vectorstore for a specific language
def train_vectorstore(language="en", force_retrain=False):
    """Train and save the vectorstore for a specific language"""
    global VECTORSTORES, EMBEDDINGS_MODEL
    
    if EMBEDDINGS_MODEL is None:
        EMBEDDINGS_MODEL = initialize_embeddings()
        if EMBEDDINGS_MODEL is None:
            return None
    
    index_path = f"faiss_{language}_index"
    
    # Check if we need to train
    if not force_retrain and os.path.exists(index_path):
        try:
            logger.info(f"Loading existing index from {index_path}...")
            # Safely load the index with dangerous deserialization allowed
            vectorstore = FAISS.load_local(index_path, EMBEDDINGS_MODEL, allow_dangerous_deserialization=True)
            logger.info(f"Successfully loaded existing vectorstore for {language}")
            VECTORSTORES[language] = vectorstore
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading existing index: {e}. Will retrain and recreate index.")
            # Remove potentially corrupted index file
            try:
                os.remove(index_path)
            except Exception as remove_error:
                logger.warning(f"Could not remove corrupted index file: {remove_error}")
    
    # Need to train a new model
    logger.info(f"Creating new vectorstore for language: {language}...")
    docs = load_faq_documents(language)
    
    if not docs:
        logger.error(f"ERROR: No documents found for language: {language}")
        return None
    
    # Retry mechanism for embeddings model
    embeddings_models = [
        EMBEDDINGS_MODEL,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    ]
    
    for embeddings_model in embeddings_models:
        try:
            vectorstore = FAISS.from_documents(docs, embeddings_model)
            
            # Ensure index directory exists
            os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
            vectorstore.save_local(index_path)
            
            logger.info(f"Successfully trained and saved vectorstore for {language} at {index_path}")
            VECTORSTORES[language] = vectorstore
            return vectorstore
        except Exception as e:
            logger.warning(f"Vectorstore training failed with model {embeddings_model}: {e}")
    
    logger.error("Failed to train vectorstore with any available embeddings model")
    return None

# Get the trained vectorstore for a language
def get_vectorstore(language="en"):
    """Get the pre-trained vectorstore for the specified language"""
    global VECTORSTORES
    
    # Return from memory if available
    if language in VECTORSTORES and VECTORSTORES[language] is not None:
        return VECTORSTORES[language]
    
    # Try to load from disk
    return train_vectorstore(language)

# Helper function to read API key from file
def read_api_key_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

# Get API key from various sources
def get_api_key():
    """Get the Groq API key from various sources"""
    # Check environment variables first
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        logger.info("Found API key in environment variables")
        return api_key
    
    # Check .env file
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        logger.info("Found API key in .env file")
        return api_key
    
    # Check file in current directory
    api_key = read_api_key_from_file('.groq_api_key')
    if api_key:
        logger.info("Found API key in .groq_api_key file")
        return api_key
    
    # Check file in home directory
    api_key = read_api_key_from_file(os.path.expanduser('~/.groq_api_key'))
    if api_key:
        logger.info("Found API key in ~/.groq_api_key file")
        return api_key
    
    logger.error("No Groq API key found")
    return None

# Verify Groq API key
def verify_groq_api_key(api_key):
    """Verify the Groq API key is valid"""
    if not api_key:
        logger.error("No API key provided for verification")
        return False
        
    import requests
    
    try:
        # Groq API endpoint for model listing
        verify_url = "https://api.groq.com/openai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Make a test request
        response = requests.get(verify_url, headers=headers, timeout=10)
        
        # Check response
        if response.status_code == 200:
            logger.info("[GROQ_API] API key verified successfully")
            return True
        else:
            logger.error(f"[GROQ_API_ERROR] API key verification failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    
    except requests.RequestException as e:
        logger.error(f"[GROQ_API_ERROR] Network error during API key verification: {e}")
        return False
    except Exception as e:
        logger.error(f"[GROQ_API_ERROR] Unexpected error during API key verification: {e}")
        return False

# Detect brand mentions in text
def detect_brand_in_text(text):
    """Detect if a non-Samsung brand is mentioned in the text"""
    # List of common electronic brands excluding Samsung (which is supported)
    other_brands = [
        "lg", "sony", "panasonic", "philips", "toshiba", "sharp", 
        "hisense", "vizio", "onida", "xiaomi", "mi", "tcl", "vu", 
        "micromax", "haier", "videocon", "whirlpool", "bosch", "ifb", 
        "godrej", "hitachi", "voltas", "electrolux", "siemens", "beko",
        "apple", "google", "oneplus", "amazon", "fire tv", "roku", "chromecast"
    ]
    
    text_lower = text.lower()
    
    # Check if any of the non-supported brands are mentioned
    for brand in other_brands:
        # Use word boundary patterns to prevent partial matches
        pattern = r'\b' + re.escape(brand) + r'\b'
        if re.search(pattern, text_lower):
            return brand
    
    # Check if Samsung is mentioned (this is our supported brand)
    if re.search(r'\bsamsung\b', text_lower):
        return "samsung"
    
    # No clear brand detected
    return None

# Get brand-specific response
def get_brand_specific_response(brand, language="en"):
    """Get a response for a non-Samsung brand inquiry"""
    if language == "en":
        return f"I apologize, but this support system is specifically designed for Samsung products only. We don't have information about {brand.title()} products. Please contact {brand.title()} customer support for assistance with their products."
    else:  # Hindi
        return f"मुझे खेद है, लेकिन यह सहायता प्रणाली केवल सैमसंग उत्पादों के लिए है। हमारे पास {brand.title()} उत्पादों के बारे में जानकारी नहीं है। कृपया उनके उत्पादों के लिए {brand.title()} ग्राहक सहायता से संपर्क करें।"

# Simple fallback response generator when LLM is not available
def generate_fallback_response(question, language="en"):
    """Generate a simple response without using the LLM when there are issues"""
    try:
        # Check if question mentions a non-Samsung brand
        detected_brand = detect_brand_in_text(question)
        if detected_brand and detected_brand != "samsung":
            return get_brand_specific_response(detected_brand, language)
            
        with open("faq_data.json", "r", encoding="utf-8") as f:
            faq_list = json.load(f)
        
        # Determine which keys to use based on language
        q_key = "question_en" if language == "en" else "question_hi"
        a_key = "answer_en" if language == "en" else "answer_hi"
        
        # Simple keyword matching
        question_lower = question.lower()
        
        # First try exact match
        for item in faq_list:
            if q_key in item and item[q_key].lower() == question_lower:
                return item[a_key]
                
        # Then try keyword match
        for item in faq_list:
            if q_key in item:
                question_words = set(question_lower.split())
                item_words = set(item[q_key].lower().split())
                
                # If more than 50% of words match
                intersection = question_words.intersection(item_words)
                if len(intersection) >= min(2, len(question_words) / 2):
                    return item[a_key]
        
        # Default fallback message
        if language == "en":
            return "I'm sorry, I don't have enough information to answer that question. Please try another question or register a complaint for direct assistance."
        else:
            return "मुझे इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है। कृपया कोई अन्य प्रश्न पूछें या सीधी सहायता के लिए शिकायत दर्ज करें।"
            
    except Exception as e:
        logger.error(f"Error in fallback response generation: {e}")
        if language == "en":
            return "I'm sorry, I'm having trouble processing your request at the moment."
        else:
            return "क्षमा करें, मुझे इस समय आपके अनुरोध को संसाधित करने में समस्या हो रही है।"

class UsageTracker:
    def __init__(self, max_requests=100, time_window=3600):
        self.request_counts = defaultdict(list)
        self.max_requests = max_requests
        self.time_window = time_window
    
    def is_allowed(self, user_id):
        current_time = time.time()
        # Remove old requests
        self.request_counts[user_id] = [
            t for t in self.request_counts[user_id] 
            if current_time - t < self.time_window
        ]
        
        # Check if within limit
        if len(self.request_counts[user_id]) < self.max_requests:
            self.request_counts[user_id].append(current_time)
            return True
        return False

usage_tracker = UsageTracker()

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract user identifier (could be IP, session ID, etc.)
        user_id = request.remote_addr
        
        if not usage_tracker.is_allowed(user_id):
            return jsonify({
                "error": "Rate limit exceeded. Please try again later.",
                "status": "429 Too Many Requests"
            }), 429
        
        return func(*args, **kwargs)
    return wrapper

def log_error_to_file(error_type, error_message, context=None):
    """Log detailed error information for later analysis"""
    try:
        with open('error_log.json', 'a') as f:
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'message': str(error_message),
                'context': context or {}
            }
            json.dump(error_entry, f)
            f.write('\n')
    except Exception as log_error:
        logger.error(f"Failed to log error: {log_error}")

# Setup QA Chain using the pre-trained vectorstore
def get_qa_chain(language="en"):
    """Create a QA chain using the pre-trained vectorstore for the specified language"""
    try:
        logger.info(f"[QA_CHAIN] Creating QA Chain for language: {language}")
        
        # Get API key
        api_key = get_api_key()
        if not api_key:
            logger.error("[QA_CHAIN_ERROR] No API key found")
            return None
        
        # Verify API key
        if not verify_groq_api_key(api_key):
            logger.error("[QA_CHAIN_ERROR] Invalid API key")
            return None
            
        logger.info("[QA_CHAIN] API key retrieved and verified successfully")
        
        # Get vectorstore
        vectorstore = get_vectorstore(language)
        if not vectorstore:
            logger.error(f"[QA_CHAIN_ERROR] Failed to get vectorstore for language: {language}")
            return None
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,  # Increased from 3 to 5 for better context
                "search_type": "similarity"  # Most similar documents
            }
        )
        
        # Initialize LLM with enhanced configuration
        try:
            llm = ChatGroq(
                api_key=api_key, 
                model_name="llama-3.3-70b-versatile", 
                temperature=0.1,  # Reduced from 0.2 for more precise answers
                max_tokens=500,  # Limit response length
                max_retries=3,  # Retry mechanism
                timeout=30  # Longer timeout for complex queries
            )
        except Exception as llm_init_error:
            logger.error(f"[LLM_INIT_ERROR] Failed to initialize LLM: {llm_init_error}", exc_info=True)
            return None
        
        # Create QA Chain with comprehensive configuration
        from langchain_core.prompts import PromptTemplate
        
        # Improved prompt template with brand-specific instructions
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a Samsung product support specialist. Only answer questions about Samsung products. 
If the question is about any other brand like LG, Sony, Xiaomi, etc., inform the user that this service is only for Samsung products.

Given the following context from Samsung's FAQ database and the user's question, provide a helpful, accurate, and concise answer:

Context:
{context}

Question: {question}

Instructions:
1. If the question is about a non-Samsung brand, politely inform the user this service only supports Samsung.
2. If the question is about Samsung but not covered in the context, say you don't have enough information.
3. If you don't know the answer, suggest registering a complaint for direct assistance.
4. Keep your answer focused and concise.

Answer:"""
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            verbose=True,  # Detailed logging
            return_source_documents=True,  # Include source context
            chain_type="stuff",  # Most direct way to use retrieved documents
            chain_type_kwargs={
                "prompt": prompt_template
            }
        )
        
        logger.info("[QA_CHAIN] QA Chain created successfully")
        return qa_chain
    except Exception as e:
        logger.error(f"[QA_CHAIN_CRITICAL_ERROR] Failed to create QA chain: {e}", exc_info=True)
        
        # Create a robust fallback
        class FallbackQAChain:
            def __init__(self, error_message, language):
                self.error_message = error_message
                self.language = language
            
            def __call__(self, query):
                question = query.get("query", "")
                fallback_response = generate_fallback_response(question, self.language)
                return {
                    "result": fallback_response,
                    "source_documents": [],
                    "error": self.error_message
                }
        
        return FallbackQAChain(str(e), language)

# Preload models in a separate thread
def preload_models():
    """Preload models in a background thread"""
    try:
        logger.info("Starting preload of models...")
        # Initialize embeddings
        global EMBEDDINGS_MODEL
        if EMBEDDINGS_MODEL is None:
            EMBEDDINGS_MODEL = initialize_embeddings()
        
        # Preload English vectorstore
        get_vectorstore("en")
        # Preload Hindi vectorstore 
        get_vectorstore("hi")
        
        logger.info("Model preloading completed!")
    except Exception as e:
        logger.error(f"Error during model preloading: {e}", exc_info=True)

@app.route('/api/faq', methods=['POST'])
@rate_limit
def handle_faq():
    """Handle FAQ requests by querying the pre-trained models"""
    start_time = time.time()
    # Request logging
    logger.info("[REQUEST] Received FAQ request")
    
    # Validate request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Auto-detect language if not specified
        language = data.get("language") or data.get("lang", "en").strip().lower()
        if not language or language not in SUPPORTED_LANGUAGES:
            language = detect(question)
            # Fallback to supported languages
            language = 'en' if language not in SUPPORTED_LANGUAGES else language
        
        logger.info(f"[PROCESSING] Question: {question}, Language: {language}")
        
        # Check if question is about a non-Samsung brand
        detected_brand = detect_brand_in_text(question)
        if detected_brand and detected_brand != "samsung":
            brand_response = get_brand_specific_response(detected_brand, language)
            return jsonify({
                "response": brand_response,
                "sources": [],
                "brand_filtered": True
            })
        
        # Get QA Chain with the pre-trained vectorstore
        qa_chain = get_qa_chain(language)
        if qa_chain is None:
            logger.error("QA Chain is None - Using direct fallback")
            fallback_response = generate_fallback_response(question, language)
            return jsonify({
                "response": fallback_response,
                "sources": [],
                "fallback": True
            })
        
        # Process query
        result = qa_chain({"query": question})
        
        # Extract answer
        answer = result.get('result', "Unable to generate response")
        source_docs = result.get('source_documents', [])
        
        # Format source documents for the response
        sources = []
        for doc in source_docs[:3]:  # Limit to top 3 sources
            if hasattr(doc, 'page_content'):
                sources.append(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            else:
                sources.append(str(doc))
        
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        return jsonify({
            "response": answer,
            "sources": sources
        })
    
    except Exception as e:
        logger.error(f"[ERROR] {e}", exc_info=True)
        
        # Generate fallback response
        fallback_response = generate_fallback_response(question, language)
        
        return jsonify({
            "response": fallback_response, 
            "error": str(e),
            "fallback": True
        })

def process_chat_request(question, language='en'):
    """Process chat request and generate appropriate response"""
    start_time = time.time()
    
    try:
        # Detect brand if mentioned
        detected_brand = detect_brand_in_text(question)
        
        # Check if a non-Samsung brand is mentioned
        if detected_brand and detected_brand != "samsung":
            return get_brand_specific_response(detected_brand, language)
        
        # Get QA Chain with the pre-trained vectorstore
        qa_chain = get_qa_chain(language)
        if qa_chain is None:
            logger.error("QA Chain is None - Using direct fallback")
            return generate_fallback_response(question, language)
        
        # Process query
        result = qa_chain({"query": question})
        
        # Extract answer
        answer = result.get('result', "Unable to generate response")
        
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        return answer
    
    except Exception as e:
        logger.error(f"[ERROR] {e}", exc_info=True)
        
        # Generate fallback response
        return generate_fallback_response(question, language)

@app.route('/api/faq', methods=['POST'])
def faq_endpoint():
    # Get request data
    data = request.json
    question = data.get('question', '')
    lang = data.get('lang', 'en')
    
    # Process the chat request
    response = process_chat_request(question, lang)
    
    return jsonify({'response': response})

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_endpoint():
    # Handle preflight request for CORS
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # Get request data
    data = request.json
    message = data.get('message', '')
    language = data.get('language', 'en')
    use_speech_input = data.get('use_speech_input', False)
    use_speech_output = data.get('use_speech_output', False)
    
    # If speech input is enabled, use speech recognition
    if use_speech_input:
        message = speech_to_text()
    
    # Process the chat request
    response = process_chat_request(message, language)
    
    # Optional text-to-speech
    if use_speech_output:
        text_to_speech(response, language)
    
    return jsonify({'response': response})

@app.route('/api/feedback', methods=['POST'])
def receive_feedback():
    """Collect user feedback on chatbot responses"""
    try:
        data = request.get_json()
        feedback = {
            'question': data.get('question', ''),
            'response': data.get('response', ''),
            'rating': data.get('rating', 0),
            'comment': data.get('comment', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to feedback database or file
        try:
            with open('user_feedback.json', 'a') as f:
                json.dump(feedback, f)
                f.write('\n')
            return jsonify({"status": "Feedback received, thank you!"}), 200
        except Exception as e:
            logger.error(f"Feedback collection error: {e}")
            return jsonify({"error": "Could not save feedback"}), 500
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}")
        return jsonify({"error": "Invalid feedback data"}), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "time": datetime.now().isoformat(),
        "supported_brands": SUPPORTED_BRANDS,
        "supported_languages": SUPPORTED_LANGUAGES
    })

# Retrain models endpoint (admin only)
@app.route('/api/admin/retrain', methods=['POST'])
def retrain_models():
    """Admin endpoint to force retraining of models"""
    try:
        admin_token = request.headers.get('X-Admin-Token')
        # Simple admin token check (should be more secure in production)
        if not admin_token or admin_token != os.environ.get('ADMIN_TOKEN', 'admin-secret-token'):
            return jsonify({"error": "Unauthorized"}), 401
            
        data = request.get_json() or {}
        language = data.get('language')
        
        if language and language in SUPPORTED_LANGUAGES:
            threading.Thread(
                target=train_vectorstore, 
                args=(language, True), 
                daemon=True
            ).start()
            return jsonify({"status": f"Retraining started for {language}"})
        else:
            # Retrain all supported languages
            for lang in SUPPORTED_LANGUAGES:
                threading.Thread(
                    target=train_vectorstore, 
                    args=(lang, True), 
                    daemon=True
                ).start()
            return jsonify({"status": "Retraining started for all languages"})
    except Exception as e:
        logger.error(f"Retrain endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Make sure we have the required files
    if not os.path.exists("faq_data.json"):
        logger.warning("Warning: faq_data.json not found. Please create it before using the FAQ endpoint.")
    
    # Check for API key
    api_key = get_api_key()
    if not api_key:
        logger.warning("Warning: No Groq API key found. Please provide a valid API key.")
    else:
        if verify_groq_api_key(api_key):
            logger.info("Groq API key verified successfully!")
        else:
            logger.warning("Warning: Groq API key verification failed.")
    
    # Start preloading models in background
    threading.Thread(target=preload_models, daemon=True).start()
    
    # Start the server
    app.run(debug=True, use_reloader=False)  # Disable auto-reloader to prevent model loading issues