from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Form, Request
import csv

app = FastAPI()
templates = Jinja2Templates(directory="project/app/templates")

# Example data model
class PredictionRequest(BaseModel):
    game_name: str

# Example prediction endpoint
@app.post("/predict/")
async def predict(data: PredictionRequest):
    # Mock prediction logic
    popularity_score = len(data.game_name) * 10  # Example calculation
    return {"game_name": data.game_name, "popularity_score": popularity_score}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):

    # Render 'index.html' with optional variables
    return templates.TemplateResponse("index.html", {"request": request, "message": "Welcome to FastAPI!"})

# CSV file to store feedback
CSV_FILE = "feedback.csv"

# Create the CSV file if it doesn't exist and add headers
with open(CSV_FILE, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Email", "Feedback"])

@app.get("/feedback", response_class=HTMLResponse)
async def feedback_form(request: Request):
    """
    Display the feedback form.
    """
    return templates.TemplateResponse("feedback_form.html", {"request": request})

@app.post("/submit/")
async def submit_feedback(
    request: Request,  # Include request here
    name: str = Form(...),
    email: str = Form(...),
    feedback: str = Form(...),
):
    """
    Handle the feedback form submission and save data to a CSV file.
    """
    # Save the feedback to the CSV file
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, email, feedback])

    # Return a success message on the feedback form page
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "Thank you for your feedback!"
    })
