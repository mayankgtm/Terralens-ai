TerraLens AI

AI-powered assistant for satellite imagery analysis

TerraLens AI helps users explore satellite imagery, ask questions about observed scenes, and visualize candidate land-cover regions with estimated geographic area statistics.

Problem Statement

Satellite imagery contains valuable information about the Earth's surface, but interpreting it can require specialized tools and remote-sensing knowledge.

TerraLens AI aims to make satellite image exploration more accessible through AI-assisted interpretation, visual analysis, and downloadable reports.

Features

- **AI Image Analysis:** Ask questions about uploaded satellite images using Gemini.
- **Satellite Imagery Retrieval:** Retrieve available Sentinel-2 imagery through Copernicus.
- **Candidate Region Detection:** Highlight potential water, vegetation, built-up, and bright-surface regions.
- **Area Estimation:** Calculate approximate geographic areas when satellite metadata is available.
- **PDF Reports:** Generate downloadable analysis reports.

Technology Stack

- Python
- Gradio
- Google Gemini API
- Copernicus Sentinel Hub
- OpenCV
- NumPy
- Pillow

Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies.
4. Configure the required Gemini and Copernicus credentials as environment variables.
5. Run `app.py`.

Never commit API keys, client secrets, or `.env` files.

Limitations

The current region detector uses HSV color-based heuristics rather than a trained land-cover segmentation model. Its classifications may confuse visually similar surfaces.

Geographic area values are approximate estimates based on the requested bounding box and image dimensions. They should not be treated as verified geospatial measurements.

## 🌍 Vision

Make Earth-observation imagery easier to explore, understand, and analyze through an accessible AI-powered interface.
