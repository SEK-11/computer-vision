import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
app = FastAPI()

# Mount static files to serve the "static" folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html on the root URL
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html") as f:
        return f.read()
# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(device)

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 4:
        im = im[:, :, :3]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...)):
    try:
        # Load the uploaded image
        image = Image.open(file.file).convert("RGBA")  # Change to RGBA to support transparency
        orig_im = np.array(image)
        orig_im_size = orig_im.shape[:2]
        model_input_size = [512, 512]

        # Preprocess
        image_tensor = preprocess_image(orig_im, model_input_size).to(device)
        result = model(image_tensor)

        # Postprocess
        result_image = postprocess_image(result[0][0], orig_im_size)

        # Create the result image (RGBA with transparency mask)
        pil_im = Image.fromarray(result_image, 'L')  # Convert to 'L' mode (grayscale)

        # Debugging: Check if the result_image has values that can act as a mask
        print("Result Image shape:", result_image.shape)
        print("Max value in result image:", np.max(result_image))
        print("Min value in result image:", np.min(result_image))

        # Apply the mask to the original image
        image = Image.open(file.file).convert("RGBA")
        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
        no_bg_image.paste(image, mask=pil_im)  # Applying the mask to the image

        # Generate a unique file name
        output_filename = f"static/output_{uuid.uuid4().hex}.png"

        # Save the image explicitly as PNG with transparency
        no_bg_image.save(output_filename, format="PNG")
        print(f"Image saved as: {output_filename}")

        # Return the download URL
        return {"download_url": f"/{output_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
