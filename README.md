# Hybrid Spatial OCR Engine (Paddle + TrOCR)

A production-grade, layout-agnostic document extraction engine designed for high-precision data capture from complex ID cards and driving licenses. This system combines the speed of **PaddleOCR** for text detection with the state-of-the-art accuracy of **Vision Transformers (TrOCR)** for text recognition.

## 🚀 Technology Stack

- **Backend**: FastAPI (Python)
- **Text Detection**: PaddleOCR (Detector-only mode)
- **Text Recognition**: TrOCR (`microsoft/trocr-small-printed`) via HuggingFace Transformers
- **Compute**: PyTorch (CUDA supported for GPU acceleration)
- **Image Processing**: OpenCV & PIL (CLAHE contrast enhancement, Cubic upscaling, and Intelligent Padding)
- **Spatial Logic**: Custom Row-Clustering and Proximity-Based Field Mapping

## 🛠 Advanced Features

### 1. Hybrid OCR Architecture
Unlike standard OCR pipelines that rely on a single model, this engine uses a **Detection-then-Transformer** workflow:
- **Paddle Detector**: Rapidly locates every text region on the page.
- **TrOCR Recognizer**: Each detected region is cropped and fed into a Vision Transformer. This resolves character-level ambiguities (e.g., mistaking '8' for 'B' or 'O' for '0') that frequently plague standard OCR engines on low-quality scans.

### 2. Intelligent Spatial Mapping
The engine does not rely on hardcoded templates. Instead, it uses **Geometric Heuristics**:
- **Row-Aware Extraction**: Groups text into logical rows to handle multi-line addresses and clustered date fields.
- **Nth-Date Logic**: Intelligently assigns Issue, NT, and TR dates based on their relative horizontal order on a single line.
- **Semantic Noise Filtering**: Automatically ignores OCR artifacts like "Holder's Signature" or "Organ Donor" labels when extracting the Holder's Name.

### 3. Production-Ready Preprocessing
- **CLAHE Enhancement**: Applies local contrast equalization to make faint text "pop" in low-light scans.
- **Adaptive Upscaling**: Small ID cards are mathematically upscaled to 2500px width to ensure the Transformer has enough pixel density for character recognition.

## 🔄 Workflow

1. **Upload**: Accepts PDF/Images via the `/scan` endpoint.
2. **Pre-Process**: Contrast enhancement and resolution normalization.
3. **Detect**: PaddleOCR identifies all bounding boxes (`det_limit_side_len=3000`).
4. **Recognize**: Each box is padded, cropped, and recognized via TrOCR.
5. **Spatial Map**:
   - Reconstructs document layout.
   - Maps labels (e.g., "Name", "DOB") to their respective values using spatial proximity.
6. **Output**: Returns a structured JSON containing validated fields and a full layout visualization.

## 📦 Requirements

- Python 3.9+
- `transformers`, `torch`, `torchvision`, `paddleocr`, `opencv-python`, `fastapi`, `uvicorn`, `Pillow`
- System dependency: `poppler-utils` (for PDF processing)

## 🏃 Running the Project

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API Server**:
   ```bash
   uvicorn main:app --reload
   ```

3. **Inference Test**:
   ```bash
   python3 test_inference.py
   ```

