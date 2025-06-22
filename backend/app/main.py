from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeEditRequest(BaseModel):
    old_resume: str
    job_description: str

editor = pipeline("text2text-generation", model="google/flan-t5-base")

@app.post("/api/edit-resume")
def edit_resume(data: ResumeEditRequest):
    prompt = (
    f"You are a professional resume editor. "
    f"Rewrite the resume below to better match the job description.\n\n"
    f"Job Description:\n{data.job_description}\n\n"
    f"Original Resume:\n{data.old_resume}\n\n"
    f"Improved Resume:"
    )
    result = editor(prompt, max_new_tokens=300)[0]['generated_text']
    return {"edited_resume": result}
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}
