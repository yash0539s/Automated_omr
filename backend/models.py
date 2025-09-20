from sqlalchemy import Column, Integer, String, JSON
from .database import Base

class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, index=True)
    student_name = Column(String, index=True)
    sheet_version = Column(String)
    subject_scores = Column(JSON)
    total_score = Column(Integer)
    json_result = Column(JSON)
    annotated_image = Column(String)
