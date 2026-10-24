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

        # Reduce minor pixel noise without heavily blurring boundaries.
        kernel = np.ones((3, 3), np.uint8)

        # HSV channels: hue [0-179], saturation [0-255],
        # value/brightness [0-255].
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]

        # 1. WATER CANDIDATES:
        # Blue/cyan colors and sufficiently saturated blue-green tones.
        water_mask = (
            (hue >= 75) &
            (hue <= 130) &
            (saturation >= 35) &
            (brightness >= 35)
        )

        # 2. VEGETATION CANDIDATES:
        # Green-dominant colors. Some green roofs and fields may be confused.
        vegetation_mask = (
            (hue >= 25) &
            (hue <= 85) &
            (saturation >= 35) &
            (brightness >= 25)
        )

        # 3. BUILT-UP CANDIDATES:
        # Relatively low-saturation or reddish/brown pixels.
        # This is a weak color heuristic, not reliable building recognition.
        built_color_mask = (
            (
                (saturation <= 75) &
                (brightness >= 45) &
                (brightness <= 225)
            ) |
            (
                (hue <= 18) &
                (saturation >= 25) &
                (brightness >= 45)
            ) |
            (
                (hue >= 170) &
                (saturation >= 25) &
                (brightness >= 45)
            )
        )

        # 4. ROAD CANDIDATES:
        # Bright, low-saturation pixels.
        # Clouds, bare ground, concrete and rooftops can also match.
        road_color_mask = (
            (saturation <= 45) &
            (brightness >= 130)
        )

        # Prioritize water, then vegetation, then built-up and road candidates.
        # This avoids painting the same pixel as several different classes.
        assigned = np.zeros(rgb.shape[:2], dtype=bool)

        masks = []
        for raw_mask in (
            water_mask,
            vegetation_mask,
            built_color_mask,
            road_color_mask,
        ):
            binary = (raw_mask.astype(np.uint8) * 255)

            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, kernel
            )
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel
            )

            candidate = binary > 0
            candidate &= ~assigned

            masks.append(candidate)
            assigned |= candidate

        water, vegetation, built_up, roads = masks

        # Transparent-looking color blending preserves the original image.
        overlay = rgb.copy()

        # Colors: water=blue, vegetation=green,
        # built-up=red, roads=yellow.
        classes = [
            (water, (0, 100, 255)),
            (vegetation, (0, 220, 70)),
            (built_up, (255, 55, 55)),
            (roads, (255, 220, 0)),
        ]

        for class_mask, color in classes:
            overlay[class_mask] = (
                0.55 * rgb[class_mask] +
                0.45 * np.array(color, dtype=np.float32)
            ).astype(np.uint8)

        total = rgb.shape[0] * rgb.shape[1]

        def class_stats(mask):
            count = int(np.count_nonzero(mask))
            percentage = (100 * count / total) if total else 0
            return count, percentage

        water_count, water_pct = class_stats(water)
        vegetation_count, vegetation_pct = class_stats(vegetation)
        built_count, built_pct = class_stats(built_up)
        road_count, road_pct = class_stats(roads)

        classified_count = (
            water_count + vegetation_count +
            built_count + road_count
        )
        classified_pct = (
            100 * classified_count / total if total else 0
        )

        report = (
            "TERRALENS AI — CANDIDATE REGION REPORT\n\n"
            f"Image dimensions: {image.width} x {image.height} pixels\n\n"
            "CANDIDATE CLASSIFICATIONS\n"
            f"Water: {water_count:,} pixels ({water_pct:.2f}%)\n"
            f"Vegetation: {vegetation_count:,} pixels "
            f"({vegetation_pct:.2f}%)\n"
            f"Built-up/color candidates: {built_count:,} pixels "
            f"({built_pct:.2f}%)\n"
            f"Road/bright-surface candidates: {road_count:,} pixels "
            f"({road_pct:.2f}%)\n\n"
            f"Total classified: {classified_pct:.2f}%\n"
            f"Unclassified: {100 - classified_pct:.2f}%\n\n"
            "INTERPRETATION AND LIMITATIONS\n"
            "Classes are assigned using RGB-derived HSV color rules, "
            "not a trained land-cover model. Water and vegetation may "
            "be confused with other surfaces. Built-up candidates are "
            "not confirmed buildings, and road candidates are not "
            "confirmed roads. These percentages represent pixel "
            "fractions, not verified geographic area. Review the "
            "overlay against the original image before drawing conclusions."
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
def fetch_satellite_image(latitude, longitude, start_date, end_date):
    """Fetch Sentinel-2 imagery for the selected location and date range."""
    import os
    import requests
    from datetime import datetime
    from PIL import Image
    from io import BytesIO

    try:
        if latitude is None or longitude is None:
            return None, "Please enter latitude and longitude."

        latitude = float(latitude)
        longitude = float(longitude)

        if not -90 <= latitude <= 90:
            return None, "Latitude must be between -90 and 90."

        if not -180 <= longitude <= 180:
            return None, "Longitude must be between -180 and 180."

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start > end:
            return None, "Start date must be before or equal to end date."

        client_id = os.getenv("COPERNICUS_CLIENT_ID")
        client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

        if not client_id or not client_secret:
            return None, (
                "Copernicus credentials are missing. Restart the app "
                "from the PowerShell session where you set them."
            )

        token_url = (
            "https://identity.dataspace.copernicus.eu/"
            "auth/realms/CDSE/protocol/openid-connect/token"
        )
        process_url = "https://sh.dataspace.copernicus.eu/process/v1"

        token_response = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
        )
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]

        # Approximate area around the chosen coordinates.
        # Longitude distance varies with latitude.
        lat_radius = 0.025
        lon_radius = min(
            0.025 / max(abs(__import__("math").cos(
                __import__("math").radians(latitude)
            )), 0.2),
            0.1,
        )

        bbox = [
            longitude - lon_radius,
            max(-90, latitude - lat_radius),
            longitude + lon_radius,
            min(90, latitude + lat_radius),
        ]

        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04"],
                output: {
                    bands: 3,
                    sampleType: "AUTO"
                }
            };
        }

        function evaluatePixel(sample) {
            return [
                2.5 * sample.B04,
                2.5 * sample.B03,
                2.5 * sample.B02
            ];
        }
        """

        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {
                        "crs": (
                            "http://www.opengis.net/def/crs/"
                            "OGC/1.3/CRS84"
                        )
                    },
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": start_date + "T00:00:00Z",
                                "to": end_date + "T23:59:59Z",
                            },
                            "mosaickingOrder": "leastCC",
                        },
                    }
                ],
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/png"},
                    }
                ],
            },
            "evalscript": evalscript,
        }

        response = requests.post(
            process_url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
            timeout=120,
        )

        if not response.ok:
            return None, (
                f"Satellite request failed ({response.status_code}): "
                f"{response.text[:800]}"
            )

        satellite_image = Image.open(BytesIO(response.content)).convert("RGB")

        return satellite_image, (
            "Satellite imagery retrieved successfully. "
            f"Location: {latitude:.5f}, {longitude:.5f}. "
            f"Requested dates: {start_date} to {end_date}. "
            "The returned image is an available satellite observation, "
            "not a live image. Check cloud cover and image quality "
            "before interpreting the results."
        )

    except ValueError:
        return None, "Enter valid coordinates and dates in YYYY-MM-DD format."
    except requests.RequestException as exc:
        return None, f"Copernicus API request failed: {exc}"
    except Exception as exc:
        return None, f"Could not retrieve satellite imagery: {exc}"
with gr.Blocks(title="TerraLens AI") as demo:
    gr.Markdown("## 🛰️ Satellite Image Explorer")
    gr.Markdown(
        "Retrieve available Sentinel-2 imagery for a location "
        "and send it to the existing analysis tools."
    )

    with gr.Row():
        satellite_latitude = gr.Number(
            label="Latitude",
            value=28.6139,
            precision=5,
        )
        satellite_longitude = gr.Number(
            label="Longitude",
            value=77.2090,
            precision=5,
        )

    with gr.Row():
        satellite_start_date = gr.Textbox(
            label="Start date (YYYY-MM-DD)",
            value="2026-09-01",
        )
        satellite_end_date = gr.Textbox(
            label="End date (YYYY-MM-DD)",
            value="2026-10-24",
        )

    fetch_satellite_button = gr.Button(
        "🌍 Fetch Satellite Image",
        variant="primary",
    )

    satellite_status = gr.Markdown()
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
    
    fetch_satellite_button.click(
        fn=fetch_satellite_image,
        inputs=[
            satellite_latitude,
            satellite_longitude,
            satellite_start_date,
            satellite_end_date,
        ],
        outputs=[
            image_input,
            satellite_status,
        ],
    )

if __name__ == "__main__":
    demo.launch()