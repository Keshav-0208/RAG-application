# **Product Comparison RAG Assistant**

This project is a complete Retrieval Augmented Generation (RAG) application built to compare products based on provided documents. It includes a chat interface that uses the Groq API for fast responses and a built-in dashboard for evaluating the system's performance.

## **Core Features**

* **Conversational Chat Interface:** Ask questions in a natural, multi-turn conversation. The application remembers the previous parts of the chat to provide contextually relevant answers.  
* **Custom Knowledge Base:** The application learns from the .txt documents you provide in the docs folder.  
* **Fast Responses with Groq:** Uses the Groq API and Llama 3 models to generate answers very quickly.  
* **Integrated Evaluation Dashboard:** The sidebar includes a dashboard that uses the RAGAS framework to measure the quality of the RAG system.  
* **Save and Load Conversations:** You can save important chat sessions and load them again later.

## **How It Works**

The application follows a standard three-step RAG process:

1. **Indexing Documents:**  
   * When the application starts, it reads all .txt files from the /docs directory.  
   * The text from these files is divided into smaller pieces using a **Sliding Window** chunking method to ensure context is preserved across chunks.  
   * An embedding model (BAAI/bge-base-en-v1.5) converts these chunks into numerical vectors, which are like fingerprints for the text.  
   * These vectors are stored in a local FAISS database for very fast searching. This process is cached, so it only happens once at startup.  
2. **Retrieving Information:**  
   * When you ask a question, your query is also converted into a vector.  
   * The application then searches the FAISS database to find the text chunks with the most similar vectors (fingerprints) to your question.  
3. **Generating an Answer:**  
   * The most relevant text chunks (the context), your original question, and the previous chat history are combined into a single prompt.  
   * This prompt is sent to the Llama 3 model via the Groq API, which generates a final answer based on all the provided information.

## **Evaluating Performance with RAGAS**

A key part of this project is the ability to test the system's quality. The sidebar contains a complete evaluation tool.

**How to Run an Evaluation:**

1. Add your own questions and ideal "ground truth" answers to the src/eval\_data.json file.  
2. Click the **"Run RAGAS Evaluation"** button in the sidebar.  
3. The application will then test your questions and grade the results using four important metrics:  
   * **Faithfulness:** Does the answer come only from the provided documents, or is it making things up?  
   * **Answer Correctness:** Is the answer correct when compared to the ideal answer you provided?  
   * **Context Recall:** Did the search find all the information needed to give a complete answer?  
   * **Context Precision:** Was the searched information relevant to the question, or was it mostly unrelated?

The results are shown in a table, giving you a clear report on the system's performance.

## **Setup and Installation**

Follow these steps to run the application on your local machine.

**1\. Clone the Repository:**

git clone \<your-repository-url\>  
cd \<your-repository-name\>

2\. Create a Virtual Environment:  
Using a virtual environment is recommended to keep project dependencies separate.  
python \-m venv venv  
source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3\. Install Dependencies:  
Install all necessary Python libraries from the requirements.txt file.  
pip install \-r requirements.txt

**4\. Configure API Keys:**

* Create a file named config.py inside the src directory.  
* Add your Groq API key and define the embedding model:  
  \# src/config.py  
  GROQ\_API\_KEY \= "your-groq-api-key-here"  
  EMBEDDING\_MODEL\_NAME \= "BAAI/bge-base-en-v1.5"

**5\. Add Your Documents:**

* Place your own .txt files into the docs folder.

**6\. Run the Application:**

* Start the Streamlit application from your terminal.

streamlit run src/main.py

The application will open in your web browser.

## **Technology Stack**

* **Application Framework:** Streamlit  
* **LLM Provider:** Groq (Llama 3\)  
* **Evaluation Framework:** RAGAS  
* **Core Libraries:** LangChain, Datasets, Pandas

#### **RAG Pipeline Components**

* **Chunking Method:** Sliding Window (RecursiveCharacterTextSplitter)  
* **Embedding Model:** BAAI/bge-base-en-v1.5 (from Hugging Face)  
* **Vector Database:** FAISS (using an IVF index)  
* **Prompting Strategy:** Zero-shot with Dynamic Context Injection