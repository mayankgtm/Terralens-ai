import os
from pathlib import Path

import requests

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

client_id = os.getenv("COPERNICUS_CLIENT_ID")
client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

if not client_id or not client_secret:
    raise RuntimeError(
        "Copernicus credentials are missing. "
        "Run this script from the PowerShell session "
        "where you set the environment variables."
    )

print("Requesting access token...")

token_response = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    },
    timeout=30,
)
token_response.raise_for_status()

access_token = token_response.json()["access_token"]
print("Authentication successful!")

# Small area around central Delhi.
bbox = [77.18, 28.60, 77.23, 28.65]

evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "dataMask"],
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
                "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
            },
        },
        "data": [
            {
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2026-09-01T00:00:00Z",
                        "to": "2026-10-24T23:59:59Z",
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

print("Requesting satellite imagery for Delhi...")

response = requests.post(
    PROCESS_URL,
    headers={"Authorization": f"Bearer {access_token}"},
    json=payload,
    timeout=120,
)

if not response.ok:
    print("Imagery request failed:", response.status_code)
    print(response.text[:1500])
    response.raise_for_status()

output_path = Path("delhi_satellite.png")
output_path.write_bytes(response.content)

print(f"SUCCESS! Satellite image saved to: {output_path.resolve()}")
print(f"Downloaded {len(response.content):,} bytes.")