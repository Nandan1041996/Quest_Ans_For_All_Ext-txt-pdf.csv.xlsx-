import time
import os
import gc
import json
import pickle
import secrets
import pandas as pd
from dotenv import load_dotenv
from functions import get_chain,get_answer
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader,TextLoader,PyPDFLoader,UnstructuredExcelLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from flask import Flask,request, render_template, redirect, url_for, flash, jsonify

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)  # Required for flash messages
os.environ['CURL_CA_BUNDLE'] = ''
# Upload folder configuration
UPLOAD_FOLDER = 'Document/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['store_answer_feedback'] = 'Store_Ans'

answer_file_path = os.path.join(app.config['store_answer_feedback'], 'answers.json')
feedback_file_path = os.path.join(app.config['store_answer_feedback'], 'feedback.json')

if not os.path.exists(answer_file_path):
    with open(answer_file_path, 'w') as f:
        json.dump([], f)  # Initialize with an empty list

def get_files_in_folder(folder_path):
    # Allowed file extensions
    allowed_extensions = ('.csv', '.pdf', '.xlsx', '.txt')
    
    # Get all files with the allowed extensions
    files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(allowed_extensions)]
    del[allowed_extensions]
    gc.collect()
    return files

@app.route('/')
def index():
    """Render the homepage with a list of allowed files."""
    # Allowed file extensions
    allowed_extensions = ('.csv', '.pdf', '.xlsx', '.txt')
    # Get the list of files with allowed extensions
    files = [os.path.basename(f) for f in get_files_in_folder(app.config['UPLOAD_FOLDER']) if f.endswith(allowed_extensions)]
    del[allowed_extensions]
    gc.collect()
    return render_template('index.html', files=files)

@app.route('/uploads', methods=['POST'])
def upload_file():
    """Handle file uploads."""
    allowed_extensions = ('.csv', '.pdf', '.xlsx', '.txt')
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith(allowed_extensions):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File uploaded successfully')
    else:
        flash(f'Invalid file type. Only {", ".join(allowed_extensions)} files are allowed.')
    del[allowed_extensions,file]
    gc.collect()
    return redirect(url_for('index'))

@app.route('/delete/<filename>')
def delete_file(filename):
    """Delete a file.
    Args:
    filename(['String']) : Path of file to be deleted.
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.remove(file_path)
    flash('File deleted successfully.')
    pkl_file_name = filename.split('.')[0]+'.pkl'
    if os.path.exists(os.path.join(pkl_file_name)):                
        os.remove(os.path.join(pkl_file_name))
    del [file_path,pkl_file_name]
    gc.collect()
    return redirect(url_for('index'))

# general prompt for model
prompt_temp = '''You are an SAP Guide AI. Your role is to provide guidance strictly on SAP-related topics based on the given context only and provide answer in proper and intractive way. 
and before providing answer go through all the rules and try to find which rule will be best for related question.Do not use unnecessary information in output and try to provide synthesized answer.
**Rules:**

1  One Word in Question:
   If a question contains only one words find the number of questions that contain the specific word(s). Then,\
   combine all the questions containing the specific word(s) and provide the answer based on the combined context.
    Process:
        1. find how many questions contain the word(s) and provide a list of all those questions. Combine their context to generate a synthesized answer.
        2. Separate each question with a new line in the provided response.
        3. If no question contains the word(s), respond with: "Specific word-related information not available."

2. Two Word in Question:
   If a question contains only Two words and if a relevant question is found containing those words, provide the answer for that question. If no such question exists,\
   find the number of questions that contain the specific word(s). Then, combine all the questions containing the specific word(s) and provide the answer based on the combined context.
    Process:
        1. Search for the exact word(s) in the of questions.
        2. If a relevant question is found, return the associated answer.
        3. If no exact match is found, count how many questions contain the word(s) and provide a list of all those questions. Combine their context to generate a synthesized answer.
        4. Separate each question with a new line in the provided response.
        5. If no question contains the word(s), respond with: "Specific word-related information not available."

3. Three Word in Question:
   If a question contains only Three words and if a relevant question is found containing those words, provide the answer for that question. If no such question exists,\
   find the number of questions that contain the specific word(s). Then, combine all the questions containing the specific word(s) and provide the answer based on the combined context.
    Process:
        1. Search for the exact word(s) in the of questions.
        2. If a relevant question is found, return the associated answer.
        3. If no exact match is found, count how many questions contain the word(s) and provide a list of all those questions. Combine their context to generate a synthesized answer.
        4. Separate each question with a new line in the provided response.
        5. If no question contains the word(s), respond with: "Specific word-related information not available."

4. **Use SAP Context Only:** 
   Answer only from the given SAP-specific context. If a question is about non-SAP tools (e.g., ChatGPT or web apps etc.),\
   respond with: "Please ask question related to SAP."

5. Please check if the following question uses the exact or similar concepts. If the concepts match and you can\
   confidently answer based on that, then provide an answer. If the concepts do not align or you are unsure, respond with 'Answer is not found in provided context.Provide more information. .'" or \
   If the key term aligns with the intended meaning, answer the question. If the key term does not match the intended meaning, say 'Answer is not found in provided context.Provide more information.'"
    Example:
    "What is a maintenance plant list?"
    "What is a maintenance plan list?"

5a. Exact Match with Variations:
    Search in the vector database using only the Question field. Ensure your response fully matches the question in meaning, even if slight variations are present\
    (e.g., adding phrases befor or after question like "related to sap", "in sap",regards to sap", "regarding to sap", "for sap," "within sap," "about sap").\
    double or trible check If the core question is related to sap and possible that answer can found within provided context then proceed with the answer. If the user's question, despite \
    these variations, does not match any question in the provided context, respond with: \
    "The answer for this question is not available in the provided context."
    ex. what is maintenance plant list regarding to sap 
        what is maintenance plant list for planning or plan etc. reply "Information About Question Is Not Available In Given Context."

6. Irrelevant or Misleading Variations:
    Search in the vector database using only the Question field. If the user's question includes additional phrases or variations (before or after the main question) that are not related to SAP \
    or the provided context, respond with: "Please ask SAP-related questions from provided information." This applies even if the core of the question seems related but the added \
    variations make it irrelevant to SAP or the context.

7. **Unrelated or Confusing Questions:** 
   Search in the vector database using only the Question field. If the question is unrelated to SAP, or unclear, or contains confusing parts, respond with: "Please provide more information related to question."

8. **Word Confusion:**
   Search in the vector database using only the Question field. If a question has words that don't match the context (e.g., "maintenance plant list" or plan list instead of "maintenance plan list"),\
   double-check the terms. If the context doesn't cover the exact term, respond with: "Provided More Information About Question."

9. **Greetings and Common Phrases:**
   - If a question contains greetings (e.g., "Hello, Hi"), respond with: "Hello, please ask your question."
   - If a user thanks you, reply with: "You're welcome! Feel free to ask if you have any other SAP-related questions."
   - For small talk (e.g., "How are you?"), reply with: "Please ask SAP-related questions."

10. strictly Follow: 
    do not answer the question using own knowledge or Information outside of provided context.

**Example Non-SAP Questions (Respond with "I don't know"):**
- "How do I use ChatGPT?" or how do i use cnnkdddnoalmlcbhxcaauq*?
- "What is a Pending PR List for a non-SAP system?"

**Example Questions and Responses:**
- **What can you do?** → "I provide guidance on SAP-related topics based on the provided context."
- **Are you a robot?** → "I am an AI designed to assist with SAP inquiries."
- **Where do you get your information?** → "I use SAP-specific data to generate answers."
- **Can you give me an example?** → "Sure! You can ask about topics like 'What is my release code in SAP?'"
- **Information about sap** or **tell me about sap** -> "'give information about context you know'"
Context:
{context}
Question:
{question}
'''

# embeding is representation of data into vector formate 
embedings = HuggingFaceBgeEmbeddings()
# define model and asign key # if do sample is true then it uses random sample technique and 
# llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct', temperature=0.001, huggingfacehub_api_token='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', max_new_tokens=512,do_sample=False)

@app.route('/ask', methods=['POST'])
def get_ans_from_csv():
    ''' this function is used to get answer from given csv.

    Args:
    doc_file([CSV]): comma separated file 
    query_text :  Question

    Returns: Answer
    '''
    query_text = request.form.get('query_text')
    doc_file = request.form.get('selected_file')
    print('doc_file::',doc_file)
    selected_language = request.form.get('selected_language')
    query_text = query_text.lower() 
    ext = doc_file.split('.')[1]
    prompt =  PromptTemplate(template=prompt_temp,input_variables=['context','question'])

    if query_text :
        start_time = time.time()
        if not doc_file or doc_file == "Select a document":
            flash("Please select a document to proceed.")
            return redirect(url_for('index'))
        else:
            if doc_file.endswith('txt'):
                loader = TextLoader(os.path.join('Document', doc_file), encoding='utf-8')
                data = loader.load()
            elif doc_file.endswith('csv'):
                loader = CSVLoader(os.path.join('Document',doc_file))
                prompt =  PromptTemplate(template=prompt_temp,input_variables=['context','question'])
                data = loader.load()
            elif doc_file.endswith('pdf'):
                loader = PyPDFLoader(os.path.join('Document',doc_file))
                data = []
                for page in loader.lazy_load():
                    data.append(page)
            elif doc_file.endswith(('xlsx','xls')):
                loader = UnstructuredExcelLoader(os.path.join('Document',doc_file))
                prompt =  PromptTemplate(template=prompt_temp,input_variables=['context','question'])
                data = loader.load()
                print('data::',data)

            # to load pickle file 
            pickle_file_name = doc_file.split('.')[0]+'.pkl'
            if os.path.isfile(os.path.join(pickle_file_name)):
                with open(os.path.join(pickle_file_name),mode='rb') as f:
                    vector_index = pickle.load(f)
            else:
                # if pickle file not available
                vector_index = FAISS.from_documents(data, embedding=embedings)
            # model
            llm = ChatGroq(model='llama-3.1-70b-versatile',api_key='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',temperature=0,max_retries=2)

            # function is used to get answer
            chain = get_chain(llm,prompt,vector_index,ext)
            res_dict = get_answer(chain,query_text,ext)

            if not os.path.isfile(os.path.join(pickle_file_name)):
                with open(os.path.join(pickle_file_name),mode='wb') as f:
                        pickle.dump(vector_index,f)

            # Prepare the answer dictionary
            que_ans_dict = {'doc': doc_file, query_text:res_dict}
            res_ans =que_ans_dict[query_text][selected_language]
            
            del [pickle_file_name,vector_index,chain,llm,
                res_dict,que_ans_dict,loader,data,prompt]
            gc.collect()

        end_time = time.time()
        print('time_taken :',end_time-start_time)
        return jsonify({'answer': res_ans})
    else:
        return redirect(url_for('index'))
    



@app.route('/save_answers', methods=['POST'])
def save_answers():
    """Save answer data to answers.json."""
    data = request.json
    print('answer_file_path:',answer_file_path)
    with open(answer_file_path, 'r+') as f:
        answers = json.load(f)
        answers.append(data)  # Append new data
        f.seek(0)  # Move to the beginning of the file
        json.dump(answers, f, indent=4)  # Save updated data
    del[data]
    gc.collect()
    return jsonify({'message': 'Answer data saved successfully'}), 200

@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    """Save user feedback."""
    feedback_data = request.json
    # Initialize feedback file if it doesn't exist
    if not os.path.exists(feedback_file_path):
        with open(feedback_file_path, 'w') as f:
            json.dump([], f)  # Start with an empty list
    
    print('feedback_file_path::',feedback_file_path)
    with open(feedback_file_path, 'r+') as f:
        feedbacks = json.load(f)
        feedbacks.append(feedback_data)  # Append new feedback
        f.seek(0)  # Move to the beginning of the file
        json.dump(feedbacks, f, indent=4)  # Save updated feedback

    return jsonify({'message': 'Feedback saved successfully'}), 200


if __name__=='__main__':
    app.run(debug=True,use_reloader=False)







