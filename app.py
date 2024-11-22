from flask import Flask, request, jsonify
import pdfplumber
import os
import openai
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv('GITHUB_API_KEY')
openai.api_base = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_pdf_text(pdf_path):
    """Extract text from PDF file using pdfplumber."""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.debug(f"PDF has {len(pdf.pages)} pages")
            
            # Extract text from each page
            text = []
            for i, page in enumerate(pdf.pages):
                try:
                    # Extract text from the page
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                        logger.debug(f"Page {i + 1} text length: {len(page_text)}")
                        logger.debug(f"Page {i + 1} first 200 chars: {page_text[:200]}")
                    else:
                        logger.warning(f"No text extracted from page {i + 1}")
                        
                        # Try extracting words directly
                        words = page.extract_words()
                        if words:
                            page_text = ' '.join(word['text'] for word in words)
                            text.append(page_text)
                            logger.debug(f"Extracted {len(words)} words from page {i + 1}")
                        else:
                            logger.warning(f"No words found on page {i + 1}")
                            
                            # Try extracting tables
                            tables = page.extract_tables()
                            if tables:
                                table_text = []
                                for table in tables:
                                    table_text.extend(' '.join(str(cell) for cell in row if cell) for row in table)
                                if table_text:
                                    text.append('\n'.join(table_text))
                                    logger.debug(f"Extracted {len(tables)} tables from page {i + 1}")
                            else:
                                logger.warning(f"No tables found on page {i + 1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i + 1}: {str(e)}")
                    continue
            
            # Combine all text
            final_text = '\n'.join(text)
            
            if final_text.strip():
                logger.debug(f"Total extracted text length: {len(final_text)}")
                logger.debug(f"First 500 characters: {final_text[:500]}")
                return final_text
            else:
                logger.error("No text could be extracted from the PDF")
                return None
            
    except Exception as e:
        logger.error(f"Error processing the PDF: {str(e)}")
        return None

def get_answer_from_model(question, context):
    """Get answer from GPT-4o model."""
    try:
        logger.debug("Making request to GPT-4o model")
        logger.debug(f"Context length: {len(context)} characters")
        logger.debug(f"First 500 characters of context: {context[:500]}")
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Only answer based on the information given in the context."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer the question based only on the provided context."
            }
        ]
        
        logger.debug("Sending request to model")
        
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        logger.debug(f"Response: {response}")
        
        if response and response.choices:
            answer = response.choices[0].message.content
            logger.debug(f"Generated answer: {answer}")
            return answer
        else:
            logger.error("No valid response from model")
            return None
            
    except Exception as e:
        logger.error(f"Error calling model API: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.debug(f"Processing PDF file: {filepath}")
        
        # Extract text from PDF
        text = extract_pdf_text(filepath)
        if not text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Store the text with the filename
        app.pdf_texts = getattr(app, 'pdf_texts', {})
        app.pdf_texts[filename] = text
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'text_length': len(text)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    # Check if API key is configured
    if not os.getenv('GITHUB_API_KEY'):
        return jsonify({'error': 'API key not configured'}), 500

    data = request.json
    if not data or 'question' not in data or 'filename' not in data:
        return jsonify({'error': 'Question and filename are required'}), 400
    
    filename = data['filename']
    question = data['question']
    
    # Check if we have the text for this file
    if not hasattr(app, 'pdf_texts') or filename not in app.pdf_texts:
        return jsonify({'error': 'File not found or not processed'}), 404
    
    try:
        # Get the text and get answer from model
        text = app.pdf_texts[filename]
        logger.debug(f"Processing question for file: {filename}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Text length: {len(text)} characters")
        logger.debug(f"First 500 characters of text: {text[:500]}")
        
        answer = get_answer_from_model(question, text)
        
        if answer:
            return jsonify({
                'answer': answer,
                'text_length': len(text)
            }), 200
        else:
            return jsonify({
                'error': 'Could not get an answer from the model'
            }), 500
        
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
