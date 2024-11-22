from flask import Flask, request, jsonify
import pdfplumber
import os
import openai
import logging
import fitz  # PyMuPDF
import base64
from PIL import Image
from io import BytesIO
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
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF using PyMuPDF."""
    images = []
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Iterate through pages
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get images from page
            image_list = page.get_images()
            
            # Process each image
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(BytesIO(image_bytes))
                    
                    # Save image
                    image_filename = f'page_{page_num + 1}_image_{img_index + 1}.png'
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', image_filename)
                    image.save(image_path)
                    
                    # Convert image to base64 for model input
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    images.append({
                        'filename': image_filename,
                        'page': page_num + 1,
                        'base64': img_base64
                    })
                    
                    logger.debug(f"Extracted image {img_index + 1} from page {page_num + 1}")
                except Exception as e:
                    logger.error(f"Error processing image {img_index + 1} on page {page_num + 1}: {str(e)}")
                    continue
        
        pdf_document.close()
        return images
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {str(e)}")
        return []

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

def get_answer_from_model(question, context, images=None):
    """Get answer from GPT-4o model with image support."""
    try:
        logger.debug("Making request to GPT-4o model")
        logger.debug(f"Context length: {len(context)} characters")
        logger.debug(f"Number of images: {len(images) if images else 0}")
        
        # Prepare context with both text and image descriptions
        full_context = context
        if images:
            full_context += "\n\nThe document also contains the following images:\n"
            for idx, img in enumerate(images, 1):
                full_context += f"\nImage {idx} (on page {img['page']}):\n"
                # Get image description from model
                try:
                    image_messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that describes images accurately and concisely."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img['base64']}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this image briefly and include any visible text."
                                }
                            ]
                        }
                    ]
                    
                    image_response = openai.ChatCompletion.create(
                        model=MODEL_NAME,
                        messages=image_messages,
                        max_tokens=150
                    )
                    
                    if image_response and image_response.choices:
                        img_description = image_response.choices[0].message.content
                        full_context += f"{img_description}\n"
                except Exception as e:
                    logger.error(f"Error getting image description: {str(e)}")
                    continue
        
        logger.debug(f"First 500 characters of full context: {full_context[:500]}")
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context, including both text and images. Only answer based on the information given in the context."
            },
            {
                "role": "user",
                "content": f"Context: {full_context}\n\nQuestion: {question}\n\nAnswer the question based only on the provided context."
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
        
        # Extract images from PDF
        images = extract_images_from_pdf(filepath)
        logger.debug(f"Extracted {len(images)} images from PDF")
        
        if not text and not images:
            return jsonify({'error': 'Could not extract any content from PDF'}), 400
        
        # Store the content with the filename
        app.pdf_contents = getattr(app, 'pdf_contents', {})
        app.pdf_contents[filename] = {
            'text': text or '',
            'images': images
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'text_length': len(text) if text else 0,
            'image_count': len(images)
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
    
    # Check if we have the content for this file
    if not hasattr(app, 'pdf_contents') or filename not in app.pdf_contents:
        return jsonify({'error': 'File not found or not processed'}), 404
    
    try:
        # Get the content
        content = app.pdf_contents[filename]
        text = content['text']
        images = content['images']
        
        logger.debug(f"Processing question for file: {filename}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Text length: {len(text)} characters")
        logger.debug(f"Number of images: {len(images)}")
        
        answer = get_answer_from_model(question, text, images)
        
        if answer:
            return jsonify({
                'answer': answer,
                'text_length': len(text),
                'image_count': len(images)
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
