# 📄 PDF Text Scanner (pdfscan)

A high-performance Python tool for extracting text from PDFs using **Tesseract OCR** and **OpenCV** image preprocessing.

## 🚀 Features
- **Multi-page Support**: Extracts text from every page of a PDF.
- **Advanced Preprocessing**: Uses OpenCV to denoise and binarize images for maximum OCR accuracy.
- **Editable Output**: Consolidates results into a clean `.txt` file.
- **Simple Workflow**: Drop your files in `uploads/` and get results in `output/`.

## 🛠️ Installation

### 1. System Dependencies
Ensure you have the following installed on your system (for Linux/Pop!_OS):
```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

### 2. Python Requirements
Install the necessary Python libraries:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Run the API Server
```bash
uvicorn main:app --reload
```

### Scan a PDF (API)
You can use `curl` or any API client (like Postman) to upload a file:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/scan' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/document.pdf;type=application/pdf'
```

### Documentation
Once the server is running, visit:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
