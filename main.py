import io, json, pathlib, re, sys, time, textwrap, base64
from typing import Dict, List, Tuple, Optional
import os
import uuid
import shutil
from dotenv import load_dotenv
import cv2, numpy as np, pdfplumber
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from openai import (
    OpenAI, RateLimitError, APIError, APIConnectionError, Timeout
)

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Body, BackgroundTasks, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn

# Load environment variables
load_dotenv()

# ────────────────────────────────────────────────────────────────
# 🔑  Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_KEY = os.getenv("API_KEY", "your-api-key")  # API key for securing endpoints
# ---------------------------------------------------------------

# Initialize API clients
genai.configure(api_key=GEMINI_API_KEY)
_client = OpenAI(api_key=OPENAI_API_KEY)

# Create folders for uploads and results
UPLOAD_DIR = pathlib.Path("./uploads")
QUESTION_DIR = UPLOAD_DIR / "questions"
ANSWER_DIR = UPLOAD_DIR / "answers"
RESULT_DIR = pathlib.Path("./results")

for dir_path in [UPLOAD_DIR, QUESTION_DIR, ANSWER_DIR, RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Exam Paper Grading API",
    description="API for grading exam papers using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving result files
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# =============================================================================
# 0️⃣  Retry wrapper
# =============================================================================
def backoff_openai(**kwargs) -> str:
    """Call the ChatCompletion endpoint with exponential‑backoff retries."""
    delay = 3
    for attempt in range(6):
        try:
            return (
                _client.chat.completions.create(**kwargs)
                .choices[0].message.content
            )
        except (RateLimitError, APIError, APIConnectionError, Timeout):
            if attempt == 5:
                raise
            print(f"   · OpenAI error – retrying in {delay}s")
            time.sleep(delay)
            delay *= 2


# =============================================================================
# 1️⃣  Question‑paper extractor  (GPT‑4o, JSON‑mode)
# =============================================================================
QP_MODEL        = "gpt-4o"
QP_CHUNK_TOKENS = 3_000

def _pdf_text(pdf: pathlib.Path) -> str:
    """Return all plain‑text extracted from the PDF pages (no OCR)."""
    with pdfplumber.open(pdf) as f:
        return "\n".join(p.extract_text() or "" for p in f.pages)

def _chunk(text: str, max_tokens: int = QP_CHUNK_TOKENS) -> List[str]:
    """Break large text into ~token‑sized pieces so GPT can parse."""
    char_lim = max_tokens * 4
    buf, buf_len, out = [], 0, []
    for line in text.splitlines():
        buf.append(line)
        buf_len += len(line)
        if buf_len >= char_lim:
            out.append("\n".join(buf))
            buf, buf_len = [], 0
    if buf:
        out.append("\n".join(buf))
    return out

def _extract_q_chunk(chunk: str) -> List[dict]:
    """Ask GPT‑4o to pull question‑number / text / marks triples from chunk."""
    prompt = f"""
    Extract every distinct exam question in the text below.

    Return a JSON object:
    {{
      "questions": [
        {{ "number": <string>, "question": <string>, "marks": <int|null> }}, …
      ]
    }}

    • Keep the *full* text of each question (do not truncate).
    • Preserve mathematical notation and table spacing.
    • Question "number" may include sub‑parts like 1(a), 2‑ii, etc.
    • If you cannot find the mark allocation, set "marks": null.

    Text:
    \"\"\"{chunk[:6000]}\"\"\"
    """
    raw = backoff_openai(
        model           = QP_MODEL,
        temperature     = 0,
        response_format = {"type": "json_object"},
        messages=[
            {"role": "system",
             "content": "You are an expert at parsing exam papers into structured data."},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(raw).get("questions", [])

def extract_questions(pdf: pathlib.Path) -> List[Dict]:
    """
    Return an ordered list:
        [{qid:int, question:str, marks:int}, …]
    where qid is 1‑based sequence order.
    """

    all_q: List[dict] = []
    for ch_i, ch in enumerate(_chunk(_pdf_text(pdf)), 1):
        print(f"   • GPT‑4o parsing chunk {ch_i}")
        all_q.extend(_extract_q_chunk(ch))

    merged = {q["number"]: q for q in all_q}

    def _sort_key(num: str):
        parts = re.split(r"(\d+)", num)
        return [int(p) if p.isdigit() else p for p in parts if p]

    ordered = sorted(merged.values(), key=lambda q: _sort_key(q["number"]))

    qlist = []
    for idx, q in enumerate(ordered, 1):
        qlist.append({
            "qid":      idx,
            "question": q["question"].strip(),
            "marks":    int(q["marks"]) if str(q.get("marks")).isdigit() else 0,
        })
    return qlist


# =============================================================================
# 2️⃣  OCR helpers
# =============================================================================
def _remove_shadow(img: Image.Image) -> Image.Image:
    """
    Simple shadow‑removal: dilate per‑channel → median blur → normalize diff.
    """
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    planes = cv2.split(bgr)
    cleaned = []
    for p in planes:
        dil   = cv2.dilate(p, np.ones((7, 7), np.uint8))
        bg    = cv2.medianBlur(dil, 21)
        diff  = 255 - cv2.absdiff(p, bg)
        norm  = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        cleaned.append(norm)
    return Image.fromarray(cv2.cvtColor(cv2.merge(cleaned), cv2.COLOR_BGR2RGB))

def _preprocess_for_vision(img: Image.Image) -> Image.Image:
    """
    Extra denoising before sending to the vision LLM:
      • shadow removal
      • mild bilateral filter to smooth noise but keep edges
    """
    den = _remove_shadow(img)
    cv  = cv2.cvtColor(np.asarray(den), cv2.COLOR_RGB2BGR)
    cv  = cv2.bilateralFilter(cv, 5, 75, 75)
    return Image.fromarray(cv2.cvtColor(cv, cv2.COLOR_BGR2RGB))

def _img_bytes(img: Image.Image) -> bytes:
    """JPEG‑encode image in memory."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def _gemini_ocr(jpg: bytes) -> str:
    """Call Gemini 1.5‑flash for high‑accuracy OCR."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp  = model.generate_content([{
        "role":"user",
        "parts":[
            {"text": "Extract the text exactly as shown in the image without skipping."},
            {"inline_data": {"mime_type":"image/jpeg", "data": jpg}}
        ]
    }])
    return (resp.text or "").strip()

def ocr_pdf(pdf: pathlib.Path, dpi: int = 300) -> str:
    """
    Hybrid OCR:
      • if selectable text exists, use it.
      • otherwise render to image, shadow‑clean, Gemini‑OCR it.
    """
    pages = []
    with pdfplumber.open(str(pdf)) as f:
        for i, pg in enumerate(f.pages, 1):
            native = pg.extract_text() or ""
            if native.strip():
                pages.append(native)
                continue
            img = _remove_shadow(pg.to_image(resolution=dpi).original.convert("RGB"))
            try:
                pages.append(_gemini_ocr(_img_bytes(img)))
            except Exception as e:
                pages.append(f"(⚠ OCR error p.{i}: {e})")
    

    full_text = ""
    for page_number, page in enumerate(pages, 1):
        if page_number == 1:
            full_text += f"Page {page_number} Start\n{page}"
        else:
            full_text += f"\nPage {page_number} Start\n{page}"
        full_text += f"\nPage {page_number} End"
    return full_text


# =============================================================================
# 3️⃣  Clean & sync answers  (GPT‑4o, JSON‑mode)
# =============================================================================
CLEAN_MODEL = "gpt-4o"

def clean_and_align_answers(raw_ocr: str, qlist: List[Dict]) -> List[Dict]:
    """
    Use GPT‑4o to:
      1. strip OCR noise,
      2. split into answers matched to qids.

    Returns:
        List of answers with qid, answer text, start_page, and end_page
    """
    qblock = "\n".join(f"{q['qid']}. {q['question']}" for q in qlist)

    prompt = f"""
    You are grading assistant.

    1. The text below is a noisy OCR of a student's handwritten answer script.
       It contains stray headers like "Scanned with CamScanner", page numbers,
       dates, etc. Remove *all* irrelevant noise.

    2. Split the cleaned content into **answers** that correspond to the
       questions shown in the list further below. Use your best judgement;
       if part of an answer clearly belongs to a question, attach it there.

    3. Preserve equations, ledgers or tables by either:
         • keeping exact spaces, or
         • wrapping them inside triple‑backtick blocks
           (``` … ```), whichever is clearer.

    4. Each page start and end is marked with "Page N Start" and "Page N End".
        For each answer including it's equations, ledgers, tables, all its part
        give page numbers at which the answer starts and ends in the text.
        Answers are to be returned in the order in which they appear in the text.
        If an answer starts and ends on the same page, return the same page number

    5. Output a JSON object:
       {{
         "answers": [
           {{ "qid": <int>, "answer": <string> , "start_page": <int>, "end_page": <int>}}, …
         ]
       }}

    6. If a question was *not answered*, still include its qid with
       "answer": "" (empty string).

    List of questions (for reference ONLY, do not repeat them):
    ---
    {qblock}
    ---

    No prose outside JSON.
    """
    raw_json = backoff_openai(
        model           = CLEAN_MODEL,
        temperature     = 0,
        response_format = {"type": "json_object"},
        messages=[
            {"role":"system","content":"You are an expert at cleaning OCR answer scripts."},
            {"role":"user",  "content": prompt},
            {"role":"user",  "content": f"OCR_TEXT_START\n{raw_ocr[:60_000]}\nOCR_TEXT_END"},
        ],
    )
    return json.loads(raw_json).get("answers", [])


# =============================================================================
# 4️⃣  Lenient grader  (GPT‑4o, JSON‑mode)
# =============================================================================
GRADE_MODEL = "gpt-4o"
RUBRIC = textwrap.dedent("""\
    Award marks generously.  If a student shows *some* understanding,
    give partial credit.

    Criteria:
      • Accuracy of ideas & facts
      • Coverage vs. marks available
      • Clarity / structure
      • Correct terminology
""").strip()

def grade_student(qlist: List[Dict], answers: Dict[int, str]) -> Dict:
    """
    Send (question, answer) pairs to GPT‑4o and receive per‑question scores
    and comments. Force JSON output, then clamp scores.
    """
    qa_block = []
    for q in qlist:
        qa_block.append(
            f"Q{q['qid']} ({q['marks']} marks)\n"
            f"Question: {q['question']}\n"
            f"Answer: {answers.get(q['qid'], '(No answer)')}\n"
        )
    qa_text = "\n---\n".join(qa_block)

    schema = '{ "details":[ { "qid":<int>, "score":<int>, "comment":<string> } ] }'
    reply = backoff_openai(
        model        = GRADE_MODEL,
        temperature  = 0,
        messages = [
            {"role": "system", "content": "You are a *LENIENT* but fair examiner."},
            {"role": "user",
             "content": (
                 "Grade each answer out of the mark shown.\n"
                 "Be generous where partial understanding is evident.\n"
                 f"Rubric:\n{RUBRIC}\n\n{qa_text}\n\nReturn ONLY JSON: {schema}"
             )},
        ],
        response_format = {"type": "json_object"},
    )

    try:
        raw = json.loads(reply)["details"]
    except Exception as e:
        return {
            "total_score": 0,
            "out_of":      sum(q["marks"] for q in qlist),
            "details":     [],
            "error":       f"JSON parse error: {e}",
            "raw_reply":   reply
        }

    marks_map = {q["qid"]: q["marks"]    for q in qlist}
    text_map  = {q["qid"]: q["question"] for q in qlist}

    fixed, seen = [], set()
    for d in raw:
        qid = int(d.get("qid", 0))
        if qid not in marks_map:
            continue
        score = max(0, min(int(d.get("score", 0)), marks_map[qid]))
        fixed.append({
            "qid":            qid,
            "question":       text_map[qid],
            "marks":          marks_map[qid],
            "student_answer": answers.get(qid, ""),
            "score":          score,
            "comment":        d.get("comment", "").strip()
        })
        seen.add(qid)

    for q in qlist:
        if q["qid"] not in seen:
            fixed.append({
                "qid":            q["qid"],
                "question":       q["question"],
                "marks":          q["marks"],
                "student_answer": answers.get(q["qid"], ""),
                "score":          0,
                "comment":        "Not graded by model – set to 0."
            })

    total_score = sum(d["score"] for d in fixed)
    out_of      = sum(d["marks"] for d in fixed)

    return {
        "total_score": total_score,
        "out_of":      out_of,
        "details":     sorted(fixed, key=lambda d: d["qid"])
    }


# =============================================================================
# 5️⃣  Vision‑guided placement  (GPT‑4o image input)
# =============================================================================
ANNOTATE_MODEL = "gpt-4o"

def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

_VISION_SYSTEM = (
    "You are a meticulous exam‑paper assistant. "
    "First PERFORM OCR on the page image, then locate the first line of each "
    "answer. Provide exactly one placement coordinate per visible answer."
)

def find_answer_positions(
    page_img: Image.Image,
    details: List[Dict],
    tentative_position: Dict = None
) -> Dict[int, Tuple[int, int]]:
    """
    Call GPT-4o-vision: returns {qid: (x,y)} pixel positions on page_img.
    Uses tentative_position as guidance for finding optimal positioning.
    """
    # Prepare details for the prompt
    grades_json = json.dumps([
        {"qid": d["qid"], "score": d["score"], "comment": d["comment"]}
        for d in details
    ])
    
    # Create positioning guidance based on tentative_positions if available
    position_guidance = ""
    if tentative_position:
        position_hints = []
        for d in details:
            qid = d["qid"]
            if qid in tentative_position:
                pos = tentative_position[qid]['position']
                position_hints.append(f"qid {qid}: place in {pos} section of page")
        
        if position_hints:
            position_guidance = "Position guidelines:\n• " + "\n• ".join(position_hints)
    
    prompt = textwrap.dedent(f"""
        Your tasks **in order**:
        
        1. OCR the provided page image.
        2. Find suitable EMPTY SPACES to place grade annotations.
        3. For each question ID, choose a position that:
           - Is in a clear, empty area near text
           - Won't overlap with existing content
           - Follows the position guidance when possible
        4. Respond ONLY with JSON:
           {{
             "placements":[{{"qid":<int>,"x":<int>,"y":<int>}}, …]
           }}
        
        Rules:
        • Each qid appears exactly once.
        • Coordinates are absolute pixels from the top‑left corner.
        • Choose positions with enough space for annotations.
        {position_guidance}
    """).strip()
    
    try:
        reply = backoff_openai(
            model=ANNOTATE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _b64_png(page_img), "detail": "low"},
                        },
                        {"type": "text", "text": f"GRADES = {grades_json}"}
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(reply)
        out: Dict[int, Tuple[int, int]] = {}
        for p in data.get("placements", []):
            try:
                qid = int(p["qid"])
                x = int(p["x"])
                y = int(p["y"])
                if qid not in out:  # deduplicate
                    out[qid] = (x, y)
            except (TypeError, ValueError):
                continue
        return out
    except Exception as e:
        print(f"   · Vision placement failed ({e})")
        return {}


# =============================================================================
# 6️⃣  Draw annotations
# =============================================================================
_RED = (220, 30, 25)

try:
    _FONT_BOLD   = ImageFont.truetype("arialbd.ttf", 28)
    _FONT_NORMAL = ImageFont.truetype("arial.ttf",    24)
except IOError:
    _FONT_BOLD = _FONT_NORMAL = ImageFont.load_default()

def _draw(draw: ImageDraw.ImageDraw,
          xy: Tuple[int, int],
          text: str,
          font: ImageFont.FreeTypeFont,
          pad: int = 4):
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    x, y = xy
    draw.rectangle([x-pad, y-pad, x+w+pad, y+h+pad],
                   fill=(255, 255, 255, 180))
    draw.text((x, y), text, font=font, fill=_RED)

def analyze_answer_positions(answers_data):
    """
    Analyzes answer positions on pages to determine the most likely location.
    
    Args:
        answers_data: List of answer dictionaries with qid, start_page, end_page
    
    Returns:
        Dictionary mapping each qid to its most likely page and position
    """
    # Initialize result dictionary
    result = {}
    
    # Calculate page distributions
    page_counts = {}
    for answer in answers_data:
        qid = answer.get("qid")
        start_page = answer.get("start_page")
        end_page = answer.get("end_page")
        
        # Skip invalid entries
        if not isinstance(qid, int) or not isinstance(start_page, int) or not isinstance(end_page, int):
            continue
            
        # For each answer, count which pages it appears on
        for page in range(start_page, end_page + 1):
            if page not in page_counts:
                page_counts[page] = 0
            page_counts[page] += 1
    
    # Find the most common page for answers to appear
    most_common_page = max(page_counts.items(), key=lambda x: x[1])[0] if page_counts else None
    
    # For each answer, determine its position
    for answer in answers_data:
        qid = answer.get("qid")
        start_page = answer.get("start_page")
        end_page = answer.get("end_page")
        
        # Skip invalid entries
        if not isinstance(qid, int) or not isinstance(start_page, int) or not isinstance(end_page, int):
            continue
            
        # Calculate answer span
        span_length = end_page - start_page + 1
        
        # Determine which page to focus on
        target_page = None
        position = None
        
        if span_length == 1:
            # If answer is on a single page, use that page
            target_page = start_page
            position = "middle"  # Default for single-page answers
        elif most_common_page and start_page <= most_common_page <= end_page:
            # If answer spans multiple pages but includes the most common page
            target_page = most_common_page
            
            if target_page == start_page:
                position = "top"
            elif target_page == end_page:
                position = "bottom"
            else:
                position = "middle"
        else:
            # For other multi-page answers, use heuristics
            if span_length == 2:
                # For two-page answers, prefer the first page, bottom position
                target_page = start_page
                position = "bottom"
            else:
                # For longer spans, use the middle page
                target_page = start_page + span_length // 2
                position = "middle"
        
        # Store the result
        result[qid] = {
            "page": target_page,
            "position": position,
        }
    
    return result

def annotate_script(pdf: pathlib.Path, graded: Dict, tentative_positions: Dict) -> List[pathlib.Path]:
    """
    Produce *.pageN.graded.png images with red‑ink overlays.
    Uses find_answer_positions to locate suitable empty spaces, guided by tentative_positions.
    """
    images, out_paths = [], []
    with pdfplumber.open(str(pdf)) as doc:
        for pg in doc.pages:
            raw = pg.to_image(resolution=300).original.convert("RGB")
            images.append(_preprocess_for_vision(raw))
    
    details = graded["details"]
    q_by_id = {d["qid"]: d for d in details}
    
    # Filter details for each page based on tentative_positions
    for page_idx, img in enumerate(images, 1):
        # Get questions for this page
        page_questions = [
            q_by_id[qid] for qid in tentative_positions 
            if tentative_positions[qid]['page'] == page_idx and qid in q_by_id
        ]
        
        if not page_questions:
            continue  # Skip pages with no questions
        
        draw = ImageDraw.Draw(img)
        
        # Find positions using vision model
        placements = find_answer_positions(img, page_questions, tentative_positions)
        
        # If find_answer_positions couldn't find positions, use fallback positioning
        if not placements:
            width, height = img.size
            # Create fallback positions based on tentative_positions
            for q in page_questions:
                qid = q["qid"]
                position_type = tentative_positions[qid]['position']
                
                # Set initial coordinates based on position type
                if position_type == 'top':
                    y = height * 0.15
                elif position_type == 'bottom':
                    y = height * 0.75
                else:  # 'middle'
                    y = height * 0.5
                
                x = 50  # Left margin
                placements[qid] = (int(x), int(y))
        
        # Draw annotations
        for qid, (x, y) in placements.items():
            d = q_by_id[qid]
            txt = f"{d['score']}/{d['marks']} – {d['comment']}"
            _draw(draw, (x, y), txt, _FONT_NORMAL)
        
        # Draw total on first page
        if page_idx == 1:
            _draw(draw, (50, 40),
                  f"TOTAL: {graded['total_score']}/{graded['out_of']}",
                  _FONT_BOLD, pad=6)
        
        out_file = pdf.with_name(f"{pdf.stem}.page{page_idx}.graded.png")
        img.save(out_file, format="PNG", optimize=True)
        out_paths.append(out_file)
        print(f"      ↳ saved {out_file.name}")
    
    return out_paths


# =============================================================================
# FastAPI Endpoints
# =============================================================================

# Helper functions for file handling
def save_uploaded_file(upload_file: UploadFile, directory: pathlib.Path) -> str:
    """Save an uploaded file and return its id"""
    file_id = str(uuid.uuid4())
    file_extension = pathlib.Path(upload_file.filename).suffix
    file_path = directory / f"{file_id}{file_extension}"
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    return file_id, file_path

# Route for health check
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Exam Paper Grading API is running"}

# Routes for question handling
@app.post("/api/upload-question", dependencies=[Depends(get_api_key)])
async def upload_question(file: UploadFile = File(...)):
    """Upload a question paper PDF"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id, file_path = save_uploaded_file(file, QUESTION_DIR)
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_path": str(file_path)
    }

@app.post("/api/extract-questions", dependencies=[Depends(get_api_key)])
async def extract_questions_endpoint(file_id: str = Body(...)):
    """Extract questions from a previously uploaded PDF"""
    # Find the file with the matching ID
    file_paths = list(QUESTION_DIR.glob(f"{file_id}*.pdf"))
    if not file_paths:
        raise HTTPException(status_code=404, detail="Question paper not found")
    
    file_path = file_paths[0]
    questions = extract_questions(file_path)
    
    return {
        "file_id": file_id,
        "questions": questions,
        "total_marks": sum(q["marks"] for q in questions),
        "question_count": len(questions)
    }

# Routes for answer handling
@app.post("/api/upload-answer", dependencies=[Depends(get_api_key)])
async def upload_answer(file: UploadFile = File(...)):
    """Upload an answer script PDF"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id, file_path = save_uploaded_file(file, ANSWER_DIR)
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_path": str(file_path)
    }

@app.post("/api/ocr-answer", dependencies=[Depends(get_api_key)])
async def ocr_answer(file_id: str = Body(...), dpi: int = Body(300)):
    """Perform OCR on an answer script"""
    # Find the file with the matching ID
    file_paths = list(ANSWER_DIR.glob(f"{file_id}*.pdf"))
    if not file_paths:
        raise HTTPException(status_code=404, detail="Answer script not found")
    
    file_path = file_paths[0]
    ocr_text = ocr_pdf(file_path, dpi)
    
    # Save OCR text to a file for reference
    ocr_file = RESULT_DIR / f"{file_id}.ocr.txt"
    with open(ocr_file, "w", encoding="utf-8") as f:
        f.write(ocr_text)
    
    return {
        "file_id": file_id,
        "ocr_text": ocr_text[:1000] + "..." if len(ocr_text) > 1000 else ocr_text,
        "ocr_file": str(ocr_file)
    }

@app.post("/api/clean-answers", dependencies=[Depends(get_api_key)])
async def clean_answers_endpoint(
    answer_file_id: str = Body(...),
    question_file_id: str = Body(...)
):
    """Clean and align answers with questions"""
    # Find the files
    answer_paths = list(ANSWER_DIR.glob(f"{answer_file_id}*.pdf"))
    question_paths = list(QUESTION_DIR.glob(f"{question_file_id}*.pdf"))
    
    if not answer_paths:
        raise HTTPException(status_code=404, detail="Answer script not found")
    if not question_paths:
        raise HTTPException(status_code=404, detail="Question paper not found")
    
    answer_path = answer_paths[0]
    question_path = question_paths[0]
    
    # Extract questions and perform OCR
    questions = extract_questions(question_path)
    ocr_text = ocr_pdf(answer_path)
    
    # Clean and align answers
    answers_data = clean_and_align_answers(ocr_text, questions)
    
    # Save cleaned answers to a file
    clean_file = RESULT_DIR / f"{answer_file_id}.answers.json"
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(answers_data, f, indent=2, ensure_ascii=False)
    
    return {
        "answer_file_id": answer_file_id,
        "question_file_id": question_file_id,
        "answers": answers_data,
        "answers_file": str(clean_file)
    }

@app.post("/api/grade-answer", dependencies=[Depends(get_api_key)])
async def grade_answer(
    answer_file_id: str = Body(...),
    question_file_id: str = Body(...)
):
    """Grade an answer script"""
    # Find the files
    answer_paths = list(ANSWER_DIR.glob(f"{answer_file_id}*.pdf"))
    question_paths = list(QUESTION_DIR.glob(f"{question_file_id}*.pdf"))
    
    if not answer_paths:
        raise HTTPException(status_code=404, detail="Answer script not found")
    if not question_paths:
        raise HTTPException(status_code=404, detail="Question paper not found")
    
    answer_path = answer_paths[0]
    question_path = question_paths[0]
    
    # Extract questions
    questions = extract_questions(question_path)
    
    # Check if cleaned answers exist, otherwise perform OCR and cleaning
    answers_file = RESULT_DIR / f"{answer_file_id}.answers.json"
    if answers_file.exists():
        with open(answers_file, "r", encoding="utf-8") as f:
            answers_data = json.load(f)
    else:
        ocr_text = ocr_pdf(answer_path)
        answers_data = clean_and_align_answers(ocr_text, questions)
        # Save cleaned answers
        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(answers_data, f, indent=2, ensure_ascii=False)
    
    # Convert answers_data to the format expected by grade_student
    answers_dict = {
        int(a["qid"]): a["answer"].rstrip()
        for a in answers_data if "qid" in a and a.get("answer")
    }
    
    # Grade the answers
    graded = grade_student(questions, answers_dict)
    
    # Save grading result
    grading_file = RESULT_DIR / f"{answer_file_id}.graded.json"
    with open(grading_file, "w", encoding="utf-8") as f:
        json.dump(graded, f, indent=2, ensure_ascii=False)
    
    return {
        "answer_file_id": answer_file_id,
        "total_score": graded["total_score"],
        "out_of": graded["out_of"],
        "details": graded["details"],
        "grading_file": str(grading_file)
    }

@app.post("/api/annotate-answer", dependencies=[Depends(get_api_key)])
async def annotate_answer(
    answer_file_id: str = Body(...),
    background_tasks: BackgroundTasks = None
):
    """Annotate a graded answer script"""
    # Find the files
    answer_paths = list(ANSWER_DIR.glob(f"{answer_file_id}*.pdf"))
    graded_file = RESULT_DIR / f"{answer_file_id}.graded.json"
    answers_file = RESULT_DIR / f"{answer_file_id}.answers.json"
    
    if not answer_paths:
        raise HTTPException(status_code=404, detail="Answer script not found")
    if not graded_file.exists():
        raise HTTPException(status_code=404, detail="Grading result not found")
    if not answers_file.exists():
        raise HTTPException(status_code=404, detail="Cleaned answers not found")
    
    answer_path = answer_paths[0]
    
    # Load grading result and answers
    with open(graded_file, "r", encoding="utf-8") as f:
        graded = json.load(f)
    
    with open(answers_file, "r", encoding="utf-8") as f:
        answers_data = json.load(f)
    
    # Analyze answer positions
    tentative_positions = analyze_answer_positions(answers_data)
    
    # Annotate the script
    if background_tasks:
        # Run annotation in the background for large scripts
        result_paths = []
        
        async def annotate_background():
            nonlocal result_paths
            result_paths = annotate_script(answer_path, graded, tentative_positions)
            
        background_tasks.add_task(annotate_background)
        
        return {
            "answer_file_id": answer_file_id,
            "status": "processing",
            "message": "Annotation started in the background"
        }
    else:
        # Annotate synchronously for smaller scripts
        result_paths = annotate_script(answer_path, graded, tentative_positions)
        
        return {
            "answer_file_id": answer_file_id,
            "status": "complete",
            "image_paths": [str(p) for p in result_paths],
            "image_urls": [f"/results/{p.name}" for p in result_paths]
        }

@app.post("/api/process-all", dependencies=[Depends(get_api_key)])
async def process_all(
    question_file: UploadFile = File(...),
    answer_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Process a question paper and answer script through the entire pipeline"""
    # Upload files
    q_file_id, q_file_path = save_uploaded_file(question_file, QUESTION_DIR)
    a_file_id, a_file_path = save_uploaded_file(answer_file, ANSWER_DIR)
    
    # Function to run the entire pipeline
    async def run_pipeline():
        # 1. Extract questions
        questions = extract_questions(q_file_path)
        
        # 2. Perform OCR on answer script
        ocr_text = ocr_pdf(a_file_path)
        
        # 3. Clean and align answers
        answers_data = clean_and_align_answers(ocr_text, questions)
        
        # Save cleaned answers
        answers_file = RESULT_DIR / f"{a_file_id}.answers.json"
        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(answers_data, f, indent=2, ensure_ascii=False)
        
        # 4. Convert answers to dictionary format
        answers_dict = {
            int(a["qid"]): a["answer"].rstrip()
            for a in answers_data if "qid" in a and a.get("answer")
        }
        
        # 5. Grade the answers
        graded = grade_student(questions, answers_dict)
        
        # Save grading result
        grading_file = RESULT_DIR / f"{a_file_id}.graded.json"
        with open(grading_file, "w", encoding="utf-8") as f:
            json.dump(graded, f, indent=2, ensure_ascii=False)
        
        # 6. Analyze answer positions
        tentative_positions = analyze_answer_positions(answers_data)
        
        # 7. Annotate the script
        result_paths = annotate_script(a_file_path, graded, tentative_positions)
        
        return {
            "question_file_id": q_file_id,
            "answer_file_id": a_file_id,
            "total_score": graded["total_score"],
            "out_of": graded["out_of"],
            "image_paths": [str(p) for p in result_paths],
            "image_urls": [f"/results/{p.name}" for p in result_paths]
        }
    
    if background_tasks:
        # Run in the background for large files
        background_tasks.add_task(run_pipeline)
        
        return {
            "question_file_id": q_file_id,
            "answer_file_id": a_file_id,
            "status": "processing",
            "message": "Processing started in the background"
        }
    else:
        # Run synchronously
        return await run_pipeline()

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)