# Customer Support Chatbot

## Overview
This is an advanced, multilingual customer support chatbot designed to provide intelligent, context-aware responses to user queries. The chatbot leverages state-of-the-art natural language processing technologies to deliver accurate and helpful support.
![image](https://github.com/user-attachments/assets/7392be24-d2fe-41b8-9785-32523fcf869b)
![image](https://github.com/user-attachments/assets/084a512c-276d-47bc-9e11-5e3ad52adb86)
![image](https://github.com/user-attachments/assets/4e7ff85a-3ae2-4085-ac6e-fe3039790223)
![image](https://github.com/user-attachments/assets/fe83c46e-f33d-4f75-a006-16ff5454dd3c)
![image](https://github.com/user-attachments/assets/fe45bbae-f788-4081-ada7-3598220714e6)

## Features

### 🌐 Multilingual Support
- Automatic language detection
- Support for English and Hindi
- Seamless translation and context understanding

### 🤖 Intelligent Response Generation
- Context-aware answer retrieval
- Advanced semantic search using embeddings
- Fallback mechanisms for complex queries

### 🛡️ Advanced Security
- Rate limiting to prevent API abuse
- Secure API key management
- Error logging and monitoring

### 📊 Performance Tracking
- Detailed performance logging
- Response time tracking
- User feedback collection

## Technology Stack
- **Backend**: Python, Flask
- **NLP**: 
  - Sentence Transformers
  - Hugging Face Embeddings
  - FAISS Vector Search
- **LLM**: Groq API
- **Additional Libraries**: 
  - LangChain
  - langdetect
  - python-dotenv

## Prerequisites
- Python 3.10+
- pip
- CUDA-compatible GPU (optional, but recommended)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/customer-chatbot.git
cd customer-chatbot/backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the `backend` directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Configuration

### FAQ Data
- Customize `faq_data.json` to add or modify support content
- Supports multilingual FAQ entries

### Language Support
- Currently supports English and Hindi
- Easy to extend to more languages

## Running the Application

### Development Server
```bash
python app.py
```

### Production Deployment
Use a WSGI server like Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Monitoring and Logs

### Performance Logs
- `performance_log.json`: Tracks function execution times
- `error_log.json`: Detailed error information

### User Feedback
- `user_feedback.json`: Stores user ratings and comments

## API Endpoints

### `/api/faq` (POST)
Submit a customer query
```json
{
  "question": "How do I reset my password?",
  "language": "en"  // Optional
}
```

### `/api/feedback` (POST)
Submit feedback on chatbot responses
```json
{
  "question": "Original query",
  "response": "Chatbot's answer",
  "rating": 4,
  "comment": "Helpful response!"
}
```

## Customization

### Adding New Languages
1. Update `faq_data.json`
2. Add language-specific embeddings
3. Modify `get_qa_chain()` function

### Extending FAQ Knowledge
- Edit `faq_data.json`
- Retrain vectorstore using `train_vectorstore()`

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


