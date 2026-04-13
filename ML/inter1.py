import shutil
import jason
from datetime import datetime
import numpy as np
import pandas as pd
import gradio as gr
import os
from PIL import Image
from ultralytics import YOLO
import time
import zipfile
import json
from collections import Counter

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

import plotly.graph_objects as go

MODEL_PATH="/home/omen/Downloads/ML/pcb_detector/runs/detect/yolov11m/weights/best.pt"
PROCESSED_FOLDER = "processed_results"
ZIP_NAME = "report_package.zip"
CONF_THRESHOLD=0.25

model=YOLO(MODEL_PATH)

def _save_annotated(result, out_path):
    try:
        img=result.plot()
        Image.fromarray(img).save(out_path)
        return true
    except:
        return false

def save_json(path,metadata):
    with open(path,"w") as f:
        json.dump(metadata,f,indent=2)

def create_pdf(pdf_path,orig_path,annot_path,metadata):
    doc=SimpleDocTemplate
    style=getSampleStyleSheet
    story=[]
    story.append(Paragraph(f"<b>PCB defect Report - {metadata['file_name']</b>}"))