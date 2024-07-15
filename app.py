from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import subprocess
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def capture_face(name: str):
    # Dung file face_capture.py de chup mat lay data 
    subprocess.run(["python", "face_capture.py", name])

def recognize_face():
    # Dung file face_recognition.py de nhan dien
    subprocess.run(["python", "face_recognition.py"])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/New_employee", response_class=HTMLResponse)
async def new_guys(request: Request, name: str = Form(...)):
    capture_face(name)
    # RedirectResponse de quay lai trang khoi tao
    return RedirectResponse(url="/")

@app.get("/Check_in", response_class=HTMLResponse)
async def check_in(request: Request):
    recognize_face()
    # RedirectResponse de quay lai trang khoi tao
    return RedirectResponse(url="/", status_code=303)
