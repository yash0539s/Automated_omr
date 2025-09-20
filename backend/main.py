from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil, uuid, os
from sqlalchemy.orm import Session
from backend import omr_processing, models, database
import json
import cv2

# Initialize DB
models.Base.metadata.create_all(bind=database.engine)
app = FastAPI()

UPLOAD_FOLDER = "uploads/"
PROCESSED_FOLDER = "processed/"
AUDIT_FOLDER = "audit/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(AUDIT_FOLDER, exist_ok=True)

# Load answer keys (multi-version)
answer_keys = {}
for version in ["version_1", "version_2"]:
    with open(f"backend/answer_keys/{version}.json") as f:
        answer_keys[version] = json.load(f)["answers"]

@app.post("/upload/")
async def upload_omr(student_name: str = Form(...), sheet_version: str = Form(...), file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}{uuid.uuid4()}.jpg"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Preprocess
    sheet_img = omr_processing.preprocess_omr(cv2.imread(file_location))
    responses = omr_processing.detect_bubbles(sheet_img)
    result = omr_processing.score_omr(responses, answer_keys[sheet_version])
    annotated = omr_processing.annotate_sheet(sheet_img, responses)

    annotated_path = f"{PROCESSED_FOLDER}{uuid.uuid4()}_annotated.jpg"
    cv2.imwrite(annotated_path, annotated)

    # Save JSON audit
    audit_path = f"{AUDIT_FOLDER}{uuid.uuid4()}_result.json"
    with open(audit_path, "w") as f:
        json.dump({"student_name": student_name, "responses": responses, "result": result}, f)

    # Save to DB
    db: Session = database.SessionLocal()
    db_result = models.Result(
        student_name=student_name,
        sheet_version=sheet_version,
        subject_scores=result["subject_scores"],
        total_score=result["total_score"],
        json_result=result,
        annotated_image=annotated_path
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    db.close()

    return JSONResponse(result)
