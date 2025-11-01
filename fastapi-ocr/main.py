from fastapi import FastAPI, UploadFile, File
from PIL import Image
import pytesseract
import os
from google import genai
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Gemini 클라이언트 생성 (최신 방식)
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/v1/ocr/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 1️⃣ 업로드된 이미지 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # 2️⃣ OCR 수행 (한글 + 영어)
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang="kor+eng")

        # 3️⃣ 프롬프트 작성
        prompt = f"""
        다음 OCR 텍스트에서 영양제 정보를 JSON으로 추출하세요.

        출력 형식:
        {{
          "name": "",
          "brand": ""
        }}

        텍스트:
        {text}
        """

        # 4️⃣ Gemini 2.0 Flash 모델 호출
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        # 5️⃣ 결과 정리
        result = response.text.strip()

        return {
            "filename": file.filename,
            "ocr_text": text,
            "result": result
        }

    finally:
        # ✅ 6️⃣ 처리 후 파일 삭제 (성공이든 실패든 무조건 실행)
        if os.path.exists(file_path):
            os.remove(file_path)
