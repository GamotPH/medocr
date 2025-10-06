import json, re, logging, cv2
import numpy as np
from paddleocr import PaddleOCR
import google.generativeai as genai
import requests, tempfile, os


# Medicinal Inscription Hybrid Classification-Oriented (MIHCO) Model
class MIHCO:
    def __init__(self, gemini_api_key: str):
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("ppocr").propagate = False

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            rec_model_dir='models/ppocrv4_rec/en_PP-OCRv4_rec_infer',
            cls_model_dir='models/ppocrv4_cls/ch_ppocr_mobile_v2.0_cls_infer'
        )
        self.confidence_threshold = 0.65
        self.zoom_margin = 25
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    def enhance_image(self, img):
        # Step 1: CLAHE for local contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Step 2: Sharpening to enhance edges and text clarity
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        return sharpened

    def detect_medicine_url(self, image_source: str, is_url: bool = False):
        # === Step: Load image ===
        if is_url:
            try:
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("⚠️ Could not decode image from URL.")
            except Exception as e:
                raise RuntimeError(f"❌ Error fetching image from URL: {e}")
        else:
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"⚠️ Could not read image from path: {image_source}")

        # Run OCR on original image (URL or local)
        results = self.ocr.ocr(img)
        if not results or not results[0]:


            response = f"""{{
                "generic_name": "generic name not detected",
                "brand_name": "brand name not detected",
                "dosage": "dosage not detectec",
                "expiration_date": "expiration not detected",
                "batch_lot_number": "batch lot number not detected"
                }}
                """
            output = re.sub(r"^```json|^```|```$", "", response, flags=re.MULTILINE).strip()
            parsed = json.loads(output)
            return parsed
           

        # Extract bounding boxes
        boxes = [line[0] for line in results[0] if line[1][1] > self.confidence_threshold]
        if not boxes:
            raise ValueError("⚠️ No high-confidence boxes found.")

        try:
            points = np.array([pt for box in boxes for pt in box], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(points)
        except Exception as e:
            raise RuntimeError(f"❌ Error computing bounding box: {e}")

        x1 = max(x - self.zoom_margin, 0)
        y1 = max(y - self.zoom_margin, 0)
        x2 = min(x + w + self.zoom_margin, img.shape[1])
        y2 = min(y + h + self.zoom_margin, img.shape[0])
        cropped = img[y1:y2, x1:x2]

        # === Step: Enhance image and run OCR ===
        enhanced_cropped = self.enhance_image(cropped)
        cv2.imwrite("zoomed.jpg", enhanced_cropped)

        enhanced_results = self.ocr.ocr("zoomed.jpg")

        detected_words = []
        for entry in enhanced_results[0]:
            if isinstance(entry, list) and len(entry) == 2:
                _, (text, score) = entry
                detected_words.append(f"{text} ({score:.2f})")

        if not detected_words:
            print("⚠️ No words detected from enhanced image. Falling back to original cropped image OCR.")
            original_results = self.ocr.ocr(cropped)
            for entry in original_results[0]:
                if isinstance(entry, list) and len(entry) == 2:
                    _, (text, score) = entry
                    detected_words.append(f"{text} ({score:.2f})")

        if not detected_words:
            print("⚠️ No readable text extracted. Using fallback marker.")
            detected_words = ["<no text detected>"]

        prompt = f"""
        You are a medical OCR assistant.

        From the raw OCR detection below:
        1. Generic Drug names
        2. Brand Name
        3. Dosage
        4. Expiration Date
        5. Batch or Lot Number


        Guidelines:
        - If any field is unclear, missing, corrupted, or contains placeholders like "00", return "".
        - Correct OCR misreads using common patterns and ensure proper Cases and Spacing conventions.
        - For the Generic Name, return the most common generic drug name (e.g., "Paracetamol" instead of "Acetaminophen") also account for additionals like Hydro.., HCI.., and alike if such patterns are found in the detected words.
        - For the Generic Names, also Account for possibly misspelled drug names like ..lodipine which could be amlodipine. Also they may come after + signs like combinations of generic drugs. .
        - For the Dosage, Recognize formats like "300/12.5", "500mg", etc.
        - For the Expiration Date, Make sure to format it in month year in numerical format MM/YYYY. Make sure to analyze the numbers carefully
        - If decimals are likely missing (e.g., "125" when "12.5" makes sense), correct them.

        Return the result in valid JSON format only:
        {{
        "generic_name": "...",
        "brand_name": "...",
        "dosage": "...",
        "expiration_date": "...",
        "batch_lot_number": "..."
        }}

        Detected OCR Tokens:
        {detected_words}
        """
        #response = self.model.generate_content(prompt)
        #output = re.sub(r"^```json|^```|```$", "", response.text.strip(), flags=re.MULTILINE).strip()

        response = self.model.generate_content(prompt)
        output = re.sub(r"^```json|^```|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        # response = self.model.generate_content(prompt)
        # output = response.text.strip()

        # # Clean up extra formatting (like ```json ... ```)
        # output = re.sub(r"^```[a-zA-Z]*", "", output)
        # output = re.sub(r"```$", "", output).strip()

        # match = re.search(r"\{.*\}", output, re.DOTALL)
        # if match:
        #     output = match.group(0)

        print("\n🔍 Gemini Raw Output:\n", output)

        try:
            #parsed = json.loads(output)
            #print("\n🧾 Parsed JSON:")
            #print(json.dumps(parsed, indent=2))
            #return parsed

            parsed = json.loads(output)
            print("\n🧾 Parsed JSON:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))

            # # ✅ Save to file
            # with open("detected.json", "w", encoding="utf-8") as f:
            #     json.dump(parsed, f, indent=2, ensure_ascii=False)

            return parsed
        
        except json.JSONDecodeError as e:
            raise ValueError(f"⚠️ Could not parse Gemini output as JSON. Raw output was:\n{output}")


    
    def detect_medicine(self, image_path: str):
        results = self.ocr.ocr(image_path)
        if not results or not results[0]:


            response = f"""{{
                    "generic_name": " no text detected",
                    "brand_name": "",
                    "dosage": "",
                    "expiration_date": "",
                    "batch_lot_number": ""
                    }}
                    """
            output = re.sub(r"^```json|^```|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(output)
            return parsed

        img = cv2.imread(image_path)
        boxes = [line[0] for line in results[0] if line[1][1] > self.confidence_threshold]

        if not boxes:
            raise ValueError("⚠️ No high-confidence boxes found.")

        try:
            points = np.array([pt for box in boxes for pt in box], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(points)
        except Exception as e:
            raise RuntimeError(f"❌ Error computing bounding box: {e}")

        x1 = max(x - self.zoom_margin, 0)
        y1 = max(y - self.zoom_margin, 0)
        x2 = min(x + w + self.zoom_margin, img.shape[1])
        y2 = min(y + h + self.zoom_margin, img.shape[0])
        cropped = img[y1:y2, x1:x2]

        # === Step: Enhance image and run OCR ===
        enhanced_cropped = self.enhance_image(cropped)
        cv2.imwrite("zoomed.jpg", enhanced_cropped)

        # First attempt: OCR on enhanced image
        enhanced_results = self.ocr.ocr("zoomed.jpg")
        #print("\n🔬 Raw OCR Result (Enhanced):\n", enhanced_results)

        # Try extracting words from enhanced image
        detected_words = []
        for entry in enhanced_results[0]:
            if isinstance(entry, list) and len(entry) == 2:
                _, (text, score) = entry
                detected_words.append(f"{text} ({score:.2f})")

        # If nothing readable, fallback to original cropped image
        if not detected_words:
            print("⚠️ No words detected from enhanced image. Falling back to original cropped image OCR.")
            original_results = self.ocr.ocr(cropped)
            print("\n🔬 Raw OCR Result (Original):\n", original_results)
            for entry in original_results[0]:
                if isinstance(entry, list) and len(entry) == 2:
                    _, (text, score) = entry
                    detected_words.append(f"{text} ({score:.2f})")


        # If still nothing detected, insert placeholder
        if not detected_words:
            print("⚠️ No readable text extracted from either enhanced or original image. Using fallback marker.")
            detected_words = ["<no text detected>"]


        print("\n🔬Detected words \n", detected_words)   

        # === Step: Gemini Prompt ===
        prompt = f"""
        You are a medical OCR assistant.

        From the raw OCR detection below:
        1. Generic Drug names
        2. Brand Name
        3. Dosage
        4. Expiration Date
        5. Batch or Lot Number


        Guidelines:
        - If any field is unclear, missing, corrupted, or contains placeholders like "00", return "".
        - Correct OCR misreads using common patterns and ensure proper Cases and Spacing conventions.
        - For the Generic Name, return the most common generic drug name (e.g., "Paracetamol" instead of "Acetaminophen") also account for additionals like Hydro.., HCI.., and alike if such patterns are found in the detected words.
        - For the Generic Names, also Account for possibly misspelled drug names like ..lodipine which could be amlodipine. Also they may come after + signs like combinations of generic drugs. .
        - For the Dosage, Recognize formats like "300/12.5", "500mg", etc.
        - For the Expiration Date, Make sure to format it in month year in numerical format MM/YYYY. Make sure to analyze the numbers carefully
        - If decimals are likely missing (e.g., "125" when "12.5" makes sense), correct them.

        Return the result in valid JSON format only:
        {{
        "generic_name": "...",
        "brand_name": "...",
        "dosage": "...",
        "expiration_date": "...",
        "batch_lot_number": "..."
        }}

        Detected OCR Tokens:
        {detected_words}
        """

        response = self.model.generate_content(prompt)
        output = re.sub(r"^```json|^```|```$", "", response.text.strip(), flags=re.MULTILINE).strip()

        print("\n Analyzed Response:\n")
        print(output)

        try:
            parsed = json.loads(output)
            print("\n🧾 Parsed JSON:")
            print(json.dumps(parsed, indent=2))

            # ✅ Save to file
            with open("detected.json", "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            return parsed
        except json.JSONDecodeError:
            raise ValueError("⚠️ Could not parse Gemini output as JSON.")

