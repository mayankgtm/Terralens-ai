import os
import io
import tempfile
from xml.sax.saxutils import escape
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
)   

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from google import genai
from google.genai import types

def generate_pdf_report(image, question, ai_answer, detection_report, overlay):
    if image is None:
        raise gr.Error("Upload a satellite image before generating a report.")

    if not question or not question.strip():
        question = "No question provided."

    if not ai_answer or not ai_answer.strip():
        ai_answer = "AI analysis has not been completed."

    if not detection_report or not detection_report.strip():
        detection_report = "Candidate-region detection has not been completed."

    output_path = os.path.join(
        tempfile.gettempdir(),
        f"terralens_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
    )

    styles = getSampleStyleSheet()
    story = []

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
    )

    story.append(Paragraph("TerraLens AI", styles["Title"]))
    story.append(Paragraph("Satellite Image Analysis Report", styles["Heading2"]))
    story.append(Paragraph(
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1. Input Image", styles["Heading2"]))
    story.append(Paragraph(
        f"Image dimensions: {image.width} × {image.height} pixels",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.2 * cm))

    image_path = os.path.join(
        tempfile.gettempdir(),
        f"terralens_input_{datetime.now():%Y%m%d_%H%M%S}.png",
    )
    image.convert("RGB").save(image_path, format="PNG")
    story.append(PDFImage(image_path, width=15 * cm, height=8.5 * cm, kind="proportional"))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2. Analysis Question", styles["Heading2"]))
    story.append(
    Paragraph(
        escape(question.strip()).replace("\n", "<br/>"),
        styles["BodyText"],
    )
)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("3. AI Analysis", styles["Heading2"]))
    safe_answer = (
    ai_answer.strip()
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace("\n", "<br/>")
)
    story.append(Paragraph(safe_answer, styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4. Candidate-Region Detection", styles["Heading2"]))
    safe_detection = escape(detection_report.strip()).replace("\n", "<br/>")    
    story.append(Paragraph(safe_detection, styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    if overlay is not None:
        story.append(Paragraph("5. Detection Overlay", styles["Heading2"]))
        overlay_path = os.path.join(
            tempfile.gettempdir(),
            f"terralens_overlay_{datetime.now():%Y%m%d_%H%M%S}.png",
        )
        Image.fromarray(overlay).convert("RGB").save(overlay_path, format="PNG")
        story.append(PDFImage(
            overlay_path,
            width=15 * cm,
            height=8.5 * cm,
            kind="proportional",
        ))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Limitations", styles["Heading2"]))
    story.append(Paragraph(
        "AI observations may be uncertain. Candidate regions are identified "
        "using a simple color-based heuristic, not a trained segmentation model. "
        "Pixel fractions are not verified physical-area measurements. "
        "Review results before using them for real-world decisions.",
        styles["BodyText"],
    ))

    document.build(story)
    return output_path

def detect_water(image):
    if image is None:
        return None, "Please upload a satellite image first."

    try:
        rgb = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Heuristic candidate detection only; not verified water mapping.
        lower = np.array([35, 35, 30])
        upper = np.array([110, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        overlay = rgb.copy()
        candidates = mask > 0

        blue = np.zeros_like(rgb)
        blue[:] = [0, 100, 255]

        overlay[candidates] = (
            0.45 * rgb[candidates] + 0.55 * blue[candidates]
        ).astype(np.uint8)

        total = mask.size
        count = int(np.count_nonzero(mask))
        percentage = 100 * count / total if total else 0

        report = (
            "CANDIDATE REGION DETECTION\n\n"
            f"Image dimensions: {image.width} x {image.height} pixels\n"
            f"Candidate pixels: {count:,}\n"
            f"Candidate pixel fraction: {percentage:.2f}%\n\n"
            "INTERPRETATION\n"
            "Blue pixels match a simple HSV color rule. They are not "
            "confirmed water detections. Other surfaces may be "
            "misclassified. This percentage is not verified water "
            "coverage or physical area."
        )

        return Image.fromarray(overlay), report

    except Exception as exc:
        return None, f"Detection failed: {type(exc).__name__}: {exc}"


def analyze_image(image, question):
    if image is None:
        return "Please upload a satellite image.", None

    if not question or not question.strip():
        return "Please enter a question about the image.", image

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "GEMINI_API_KEY was not found. Set it in PowerShell, "
            "then restart the app.",
            image,
        )

    try:
        client = genai.Client(api_key=api_key)

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        prompt = f"""
You are TerraLens AI, an assistant for remote-sensing image analysis.

User question: {question}

Use evidence visible in the supplied image. Distinguish observations
from guesses. Do not invent coordinates, precise measurements, object
counts, or percentages. Explain uncertainty and keep the answer useful.
"""

        # Set GEMINI_MODEL to a model available to your API key.
        model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ],
        )

        answer = response.text or "The model returned an empty response."
        return answer, image

    except Exception as exc:
        return (
            f"AI analysis failed: {type(exc).__name__}: {exc}\n\n"
            "Check the API key, model availability, internet connection, "
            "and API usage limits.",
            image,
        )
with gr.Blocks(title="TerraLens AI") as demo:
    gr.Markdown(
        """
        # 🛰️ TerraLens AI
        ### Ask your satellite imagery. See the evidence.

        Upload an image to explore its visible features with AI or inspect
        candidate regions using a simple color-based heuristic.

        **Limitations:** These results are preliminary, not verified
        geospatial measurements or a trained land-cover segmentation model.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload satellite image",
                type="pil",
            )

            question_input = gr.Textbox(
                label="Ask a question",
                placeholder="Ask about water, vegetation, buildings, or changes...",
                lines=2,
            )

            analyze_button = gr.Button(
                "Analyze with AI",
                variant="primary",
            )
            detect_button = gr.Button("Detect candidate regions")

        with gr.Column():
            answer_output = gr.Markdown()
            image_output = gr.Image(label="Detection overlay")
            report_output = gr.Textbox(
                label="Detection report",
                lines=8,
                interactive=False,
            )            
            pdf_button = gr.Button("📄 Generate PDF Report")
        pdf_output = gr.File(label="Download TerraLens AI Report")

    analyze_button.click(
        fn=analyze_image,
        inputs=[image_input, question_input],
        outputs=[answer_output, image_output],
    )

    detect_button.click(
        fn=detect_water,
        inputs=[image_input],
        outputs=[image_output, report_output],
    )
    pdf_button.click(
    fn=generate_pdf_report,
    inputs=[
        image_input,
        question_input,
        answer_output,
        report_output,
        image_output,
    ],
    outputs=[pdf_output],
)

if __name__ == "__main__":
    demo.launch()