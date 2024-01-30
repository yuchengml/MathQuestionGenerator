import base64
import json
from typing import Dict


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def dump_to_json(result: Dict, json_file: str = 'result.json'):
    with open(json_file, 'w', ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
