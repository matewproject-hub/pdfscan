import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from src.pdf_processor import pdf_to_images_from_bytes
from src.ocr_engine import extract_text_from_image

app = FastAPI(title="PDF Text Scanner API")

# Ensure output directory exists
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Text Scanner API. Use /scan to upload a PDF."}

@app.post("/scan")
async def scan_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF and receive the extracted text.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Read file contents
        content = await file.read()
        base_name = os.path.splitext(file.filename)[0]
        
        # Convert PDF bytes to images
        images = pdf_to_images_from_bytes(content)
        if not images:
            raise HTTPException(status_code=500, detail="Failed to convert PDF to images.")

        extracted_data = []
        full_text = ""

        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            page_text = extract_text_from_image(image, page_num=i+1)
            
            extracted_data.append({
                "page": i + 1,
                "text": page_text
            })
            full_text += f"\n--- Page {i+1} ---\n\n{page_text}\n"

        # Save full cleaned text to a .txt file
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # Return both JSON data and a downloadable file URL
        response = {
            "filename": f"{base_name}.txt",
            "total_pages": len(images),
            "pages": extracted_data,
            "full_text": full_text.strip(),
            "download_url": f"/download/{base_name}.txt",
        }
        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error during scanning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
