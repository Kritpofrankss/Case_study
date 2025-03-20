from flask import Flask, render_template, request, jsonify
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename

# ตั้งค่าแอป Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import os
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=2000, openai_api_key=openai_api_key)

# ตั้งค่าตัวแปลงเวกเตอร์
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# ตั้งค่า Prompt Template สำหรับภาษาไทย
thai_prompt = ChatPromptTemplate.from_template(
    """
    คุณเป็นแชทบอทที่ช่วยตอบคำถามจากเอกสารที่ให้มาโดยตรง  
    โปรดใช้เฉพาะข้อมูลจากเอกสารที่ให้มาในการตอบคำถาม  
    หากไม่พบข้อมูลในเอกสาร กรุณาตอบว่า "ไม่พบข้อมูลดังกล่าวในเอกสาร"
    แต่ถ้ามันมีในเอกสารแต่ต้องใช้การวิเคราะห์ ให้ตอบว่า "ไม่ได้มีการบอกถึงในเอกสารโดยตรงแต่......"
    โดยคำตอบควรไม่สั้นเกินไปและยาวเกินไป
    
    ตอบคำถามเป็นภาษาไทยเท่านั้น
    
    ### **บริบทของเอกสาร:**  
    {context}  
    
    ### **คำถาม:**  
    {question}  
    
    ### **คำตอบ:**  
    """
)

# ตั้งค่า Prompt Template สำหรับภาษาอังกฤษ
english_prompt = ChatPromptTemplate.from_template(
    """
    You are a chatbot that helps answer questions directly from the provided documents.
    Please use only information from the provided documents to answer the question.
    If the information is not found in the document, please answer "This information is not found in the document".
    If it's in the document but requires analysis, answer "It's not directly mentioned in the document, but..."
    The asnwer should not be too short or too long
    Answer in English only.
    
    ### **Document Context:**  
    {context}  
    
    ### **Question:**  
    {question}  
    
    ### **Answer:**  
    """
)

# ตั้งค่า Prompt Template สำหรับการตอบอัตโนมัติตามภาษาคำถาม
auto_prompt = ChatPromptTemplate.from_template(
    """
    คุณเป็นแชทบอทที่ช่วยตอบคำถามจากเอกสารที่ให้มาโดยตรง  
    โปรดใช้เฉพาะข้อมูลจากเอกสารที่ให้มาในการตอบคำถาม  
    หากไม่พบข้อมูลในเอกสาร กรุณาตอบว่า "ไม่พบข้อมูลดังกล่าวในเอกสาร" (ถ้าเป็นภาษาอังกฤษให้ตอบว่า "This information is not found in the document")
    แต่ถ้ามันมีในเอกสารแต่ต้องใช้การวิเคราะห์ ให้ตอบว่า "ไม่ได้มีการบอกถึงในเอกสารโดยตรงแต่......" (ถ้าเป็นภาษาอังกฤษให้ตอบว่า "It's not directly mentioned in the document, but...")
    
    ตรวจสอบภาษาของคำถาม และตอบกลับด้วยภาษาเดียวกัน (ไทยหรืออังกฤษ)
    ถ้า  {question}  เป็นภาษาอังกฤษคำตอบต้องเป็นภาษาอังกฤษ
    
    ### **บริบทของเอกสาร:**  
    {context}  
    
    ### **คำถาม:**  
    {question}  
    
    ### **คำตอบ:**  
    """
)


# ฟังก์ชันโหลดเอกสาร PDF
def load_document(file_path):
    global vectors
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(docs, embeddings)
    return vectors

# ฟังก์ชันดึงข้อมูลจาก URL
def fetch_page_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator="\n", strip=True)
        
        # บันทึกข้อความลงไฟล์
        filename = secure_filename(url.split('//')[-1].replace('/', '_')) + '.txt'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return file_path
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

# ฟังก์ชันโหลดข้อมูลจากไฟล์ข้อความ
def load_text_document(file_path):
    global vectors
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(docs, embeddings)
    return vectors

# ฟังก์ชัน Paraphrase คำตอบ
def paraphrase_answer(raw_answer, language="auto"):
    try:
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)
        
        if language == "th":
            paraphrase_prompt = f"""
            กรุณาปรับคำตอบให้อ่านง่ายและเป็นธรรมชาติ  
            หลีกเลี่ยงการใช้คำหรือสัญลักษณ์ที่ผิดปกติ  
            ตอบให้อยู่ในรูปแบบที่เหมือนมนุษย์พูด โดยยังคงข้อมูลที่ถูกต้องตามต้นฉบับ
            โปรดใช้เฉพาะข้อมูลจากเอกสารที่ให้มาในการตอบคำถาม  
            
            ตอบเป็นภาษาไทยเท่านั้น แม้ว่าคำตอบเดิมจะเป็นภาษาอังกฤษก็ตาม
            
            ### คำตอบเดิม:  
            {raw_answer}  
            
            ### คำตอบที่ปรับปรุงแล้ว:
            """
        elif language == "en":
            paraphrase_prompt = f"""
            Please adjust the answer to be easy to read and natural.
            Avoid using unusual words or symbols.
            Answer in a human-like format while maintaining the accuracy of the original information.
            Please use only information from the provided documents to answer the question.
            
            Answer in English only, even if the original answer is in Thai.
            
            ### Original Answer:  
            {raw_answer}  
            
            ### Improved Answer:
            """
        else:  # auto - ตามภาษาของคำตอบเดิม
            paraphrase_prompt = f"""
            กรุณาปรับคำตอบให้อ่านง่ายและเป็นธรรมชาติ  
            หลีกเลี่ยงการใช้คำหรือสัญลักษณ์ที่ผิดปกติ  
            ตอบให้อยู่ในรูปแบบที่เหมือนมนุษย์พูด โดยยังคงข้อมูลที่ถูกต้องตามต้นฉบับ
            
            ตรวจสอบภาษาของคำตอบเดิม (ไทยหรืออังกฤษ) และใช้ภาษาเดียวกันในการตอบกลับ
            ถ้าคำตอบเดิมเป็นภาษาอังกฤษ ให้ตอบเป็นภาษาอังกฤษ ถ้าคำตอบเดิมเป็นภาษาไทย ให้ตอบเป็นภาษาไทย
            
            ### คำตอบเดิม:  
            {raw_answer}  
            
            ### คำตอบที่ปรับปรุงแล้ว:
            """
        
        response = llm.invoke(paraphrase_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"⚠️ Paraphrase Error: {e}")
        return raw_answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # ตรวจสอบว่าเป็นการอัพโหลดไฟล์หรือ URL
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "ไม่ได้เลือกไฟล์"})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        load_document(file_path)
        return jsonify({"message": "File uploaded and processed successfully"})
    
    elif 'url' in request.form:
        url = request.form['url'].strip()
        if not url:
            return jsonify({"error": "กรุณาระบุ URL"})
        
        file_path = fetch_page_text(url)
        if file_path:
            load_text_document(file_path)
            return jsonify({
                "message": f"อัพโหลดไฟล์ \"{os.path.basename(file_path)}\" สำเร็จแล้ว! คุณสามารถถามคำถามเกี่ยวกับเนื้อหาในไฟล์ได้เลย"
            })
        else:
            return jsonify({"error": "ไม่สามารถดึงข้อมูลจาก URL ได้"})
    
    return jsonify({"error": "ไม่พบข้อมูลที่ต้องการอัพโหลด"})

@app.route('/ask', methods=['POST'])
def ask_question():
    global vectors, memory
    if vectors is None:
        return jsonify({"error": "กรุณาอัพโหลดไฟล์ก่อนเริ่มถามคำถาม"})
    
    data = request.get_json()
    question = data.get("question")
    language = data.get("language", "auto")  # รับค่าภาษาที่เลือก (th, en, auto)
    
    if not question:
        return jsonify({"error": "ไม่มีคำถาม"})
    
    # เลือก prompt ตามภาษาที่ต้องการ
    if language == "th":
        prompt_template = thai_prompt
    elif language == "en":
        prompt_template = english_prompt
    else:  # auto
        prompt_template = auto_prompt
    
    # ใช้ prompt ที่เลือก
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectors.as_retriever(search_type="similarity", search_kwargs={"k": 6}),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    response = qa_chain({"question": question})
    answer = paraphrase_answer(response.get("answer", "ไม่พบคำตอบที่เกี่ยวข้อง"), language)
    
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)
    
    return jsonify({"answer": answer})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global vectors, memory
    # ล้างข้อมูลเวกเตอร์
    vectors = None
    # ล้างหน่วยความจำการสนทนา
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    # ล้างไฟล์ที่อัพโหลดไว้ (ตัวเลือก)
    # import shutil
    # shutil.rmtree(app.config['UPLOAD_FOLDER'])
    # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    return jsonify({"message": "ล้างประวัติการสนทนาและข้อมูลเรียบร้อยแล้ว"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
