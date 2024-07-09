import gradio as gr
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import csv
from docx import Document as DocxDocument
import fitz
import threading
from langdetect import detect

FIRST = "LLAMA2"
SECOND = "MISTRAL"
SYSTEM_PROMPT = """You are a helpful assistant that answers 
                user questions using the documents provided. 
                Your answer MUST be in {} language.
                Your answer MUST be in markdown format without 
                any prefixes like 'assistant:' 
                """
def process_file(file):
    file_extension = file.name.split(".")[-1].lower()

    if file_extension == 'txt':
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
        print("Reading TXT file")

    elif file_extension == 'csv':
        with open(file.name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            text = '\n'.join(','.join(row) for row in reader)
        print("Reading CSV file")
    
    elif file_extension == 'pdf':
        pdf_document = fitz.open(file.name, filetype=file_extension)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        pdf_document.close()
        print("Reading PDF file")

    elif file_extension == 'docx':
        docx_document = DocxDocument(file.name)
        text = ""
        for paragraph in docx_document.paragraphs:
            text += paragraph.text + "\n"
        print("Reading DOCX file")

    return [Document(text=text)]

def handle_file_upload(file, lang):

    llm_first = Ollama(model=FIRST.lower(), system_prompt=SYSTEM_PROMPT.format(lang), temperature=0.1, request_timeout=1200.0)
    print(f"{FIRST} LLM is ready")
    llm_second = Ollama(model=SECOND.lower(), system_prompt=SYSTEM_PROMPT.format(lang), temperature=0.1, request_timeout=1200.0)
    print(f"{SECOND} LLM is ready")

    documents = process_file(file)

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")
    Settings.text_splitter = text_splitter
    index = VectorStoreIndex.from_documents(
        documents, transformations=[text_splitter], embed_model=Settings.embed_model, show_progress=True
    )

    return index.as_query_engine(llm=llm_first), index.as_query_engine(llm=llm_second)

def query_engine(engine, question_input, results, index):
    results[index] = engine.query(question_input)

def document_qa(file_upload, question_input):
    lang = detect(question_input)
    first, second = handle_file_upload(file_upload, lang)

    print("Querying")

    results = [None, None]
    thread_first = threading.Thread(target=query_engine, args=(first, question_input, results, 0))
    thread_second = threading.Thread(target=query_engine, args=(second, question_input, results, 1))
    
    # Start the threads
    thread_first.start()
    thread_second.start()
    
    # Wait for both threads to complete
    thread_first.join()
    thread_second.join()
    
    return results[0], results[1]

with gr.Blocks() as demo:
    with gr.Row():
        file_upload = gr.File(label="Upload Document")
        question_input = gr.Textbox(label="Enter your question")
    
    with gr.Row():
        first_output = gr.Textbox(label=f"{FIRST} Output", interactive=False, lines=10)
        second_output = gr.Textbox(label=f"{SECOND} Output", interactive=False, lines=10)
    
    submit_button = gr.Button("Submit")
    
    submit_button.click(
        document_qa,
        inputs=[file_upload, question_input],
        outputs=[first_output, second_output]
    )

demo.launch()
