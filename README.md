# Question Answering System for WISESIGHT Dataset

## Overview
This project is a **Question Answering (QA) System** designed to extract accurate and relevant answers from the **WISESIGHT** dataset, which includes fundraising information and corporate profiles in **Thai and English**. The system leverages **Flask, LangChain, FAISS, and OpenAI's GPT-3.5-turbo** for document retrieval and response generation.

<p align="center">
  <img src="https://github.com/Kritpofrankss/Case_study/blob/main/IMG_6590.jpg?raw=true" width="300px">
  <img src="https://github.com/Kritpofrankss/Case_study/blob/main/IMG_6591.jpg?raw=true" width="300px">
</p>


## Features
- **Multilingual Support**: Answers questions in both Thai and English
- **Document Processing**: Handles PDF and text file inputs
- **Web Scraping**: Extracts textual data from web pages
- **Conversational Memory**: Maintains chat history for contextual responses
- **Interactive API**: Allows users to upload documents and ask questions via an API

## Technologies Used
- **Python**
- **Flask** (Backend API Framework)
- **LangChain** (NLP Pipeline)
- **FAISS** (Document Indexing and Retrieval)
- **HuggingFace Sentence Transformers** (Text Embeddings)
- **OpenAI GPT-3.5-turbo** (Answer Generation)
- **BeautifulSoup** (Web Scraping)

## Installation
### Prerequisites
- Python 3.8+
- pip
- OpenAI API key (set as an environment variable `OPENAI_API_KEY`)

### Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/qa-system.git
   cd qa-system
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key"  # On Windows: set OPENAI_API_KEY="your-api-key"
   ```

5. **Run the Flask application**
   ```bash
   python app.py
   ```

## Data Cleaning Process
### 1. **Document Formatting**
   - Extracts text from PDF files using **PyMuPDFLoader**
   - Reads `.txt` files using **TextLoader**
   - Fetches text from web pages using **BeautifulSoup**

### 2. **Splitting Documents into Chunks**
   - Uses **RecursiveCharacterTextSplitter** to split text into **500-character chunks**
   - Sets **overlap at 50 characters** to maintain context when splitting text

### 3. **Generating Text Embeddings**
   - Utilizes **sentence-transformers/all-MiniLM-L6-v2** for embeddings
   - Stores embeddings in **FAISS Index** for fast retrieval

### 4. **Cleaning Text Data**
   - Removes special characters such as `\n`, `\t`, `\r`
   - Eliminates metadata like page numbers
   - Filters out advertisements or redundant text from web pages

## API Usage
### 1. **Upload Document**
- **Endpoint:** `/upload`
- **Method:** `POST`
- **Description:** Uploads a PDF or text file, or fetches data from a provided URL
- **Example:**
  ```bash
  curl -X POST -F "file=@document.pdf" http://localhost:5001/upload
  ```

### 2. **Ask a Question**
- **Endpoint:** `/ask`
- **Method:** `POST`
- **Description:** Sends a question related to the uploaded document
- **Payload:** `{ "question": "What is the company's revenue?", "language": "auto" }`
- **Example:**
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the company’s revenue?", "language": "auto"}' http://localhost:5001/ask
  ```

### 3. **Clear Chat History**
- **Endpoint:** `/clear_history`
- **Method:** `POST`
- **Description:** Clears document vectors and chat history
- **Example:**
  ```bash
  curl -X POST http://localhost:5001/clear_history
  ```

## Project Structure
```
qa-system/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── uploads/              # Folder for uploaded documents
├── templates/            # HTML templates (if needed)
└── README.md             # Project documentation
```

## Challenges & Solutions
- **Thai Text Processing:** Implemented language-aware tokenization and embeddings
- **Efficient Document Retrieval:** Used FAISS for fast similarity searches
- **Ensuring Accurate Responses:** Used structured prompt engineering for GPT-3.5-turbo

## Future Improvements
- Add **support for more languages**
- Implement **feedback-based model retraining**
- Optimize **response generation speed**

## License
This project is for **WISESIGHT assignment purposes only**. Unauthorized use of the dataset is strictly prohibited.

