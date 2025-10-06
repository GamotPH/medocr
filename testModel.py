from medOCR import MIHCO

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get('/ocr')
def ococr(image_url: str = Query(...)):
#def ocr():

    ocr_tool = MIHCO(gemini_api_key="AIzaSyCnS8AQ_YZbq7IwZ_uoxPcFXPii4M6x0x0")

    #parsed_result = ocr_tool.detect_medicine("test_images/6.jpg")
    parsed_result = ocr_tool.detect_medicine_url(image_url, is_url=True)

    return JSONResponse(content=parsed_result)




