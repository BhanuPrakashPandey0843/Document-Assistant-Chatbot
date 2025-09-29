
# Document Assistant Chatbot

![Document Assistant Logo](logo.png)

[Watch Demo Video](video_of_chat_bot.mp4)

> **Note:** This project demonstrates a small portion of the Document Assistant I developed during my internship at CMPDI. The full internal implementation is not included due to confidentiality. This repository provides a simplified version for demonstration purposes only.

The **Document Assistant** is a Streamlit-based web application designed to process and interact with various document types, including PDFs, DOCX, PPTX, and plain text files. It leverages several Python libraries for document extraction, natural language processing (NLP), and interactive chat-based interactions.

---

## Features

- **File Upload and Processing:** Upload documents (PDF, DOCX, PPTX, TXT) to extract text and create vector representations.  
- **Text Extraction:** Uses PyMuPDF for PDFs, python-docx for DOCX, python-pptx for PPTX, and standard file reading for TXT files.  
- **Natural Language Processing:** Integrates SpaCy for text processing and linguistic analysis.  
- **Interactive Chat Interface:** Chat with the assistant using the processed document content.  
- **Vector Representations:** Generates vector embeddings for document chunks to support retrieval-based question answering.  
- **Asynchronous Processing:** Uses `asyncio` for efficient document processing.  

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd document-assistant
````

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install and run Ollama:**

   ```bash
   ollama run mistral
   ```

5. **Pull the Ollama model:**

   ```bash
   ollama pull mistral
   ```

6. **Run the Streamlit application:**

   ```bash
   streamlit run chat_pdf.py
   ```

7. **Open in your browser:**
   Navigate to `http://localhost:8501` to access the Document Assistant.

---

## Usage

1. **Upload File:** Drag-and-drop your document into the upload area or use the upload button.
2. **Process Document:** Click "Process Document" to extract text and generate vector representations.
3. **Chat Interface:** Interact with the processed document using the chat input.
4. **View Responses:** See AI-generated answers based on the document content.

---

## Libraries Used

* **PyMuPDF:** PDF document processing
* **python-docx:** DOCX document processing
* **python-pptx:** PPTX document processing
* **SpaCy:** NLP tasks
* **langchain & langchain_community:** Vectorization and chunk handling
* **Streamlit:** Interactive web interface
* **Ollama:** Language model for chat responses

---

## File Structure

```
document-assistant/
│
├─ src/
│   ├─ auth/
│   ├─ chat_pdf.py
│   └─ ...
├─ videos/
│   └─ video_of_chat_bot.mp4
├─ logo.png
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## Acknowledgments

* [Streamlit](https://streamlit.io/) – For building interactive web applications
* [SpaCy](https://spacy.io/) – NLP capabilities
* [PyMuPDF](https://pymupdf.readthedocs.io/) – PDF processing
* [python-docx](https://python-docx.readthedocs.io/) – DOCX processing
* [python-pptx](https://python-pptx.readthedocs.io/) – PPTX processing
* [langchain](https://www.langchain.com/) – Vectorization & text chunk handling
* [Ollama](https://ollama.com/) – Language model used for chat interactions

```

---

✅ This version:  
- Shows the logo at the top  
- Provides a clickable demo video link  
- Includes a clear note about this being a demonstration project  
- Maintains a professional, GitHub-ready structure  

---

If you want, I can also **create a `.gitignore` tailored to this project**, including the video, virtual environment, and Python artifacts, so you can push this repo cleanly.  

Do you want me to do that next?
```
