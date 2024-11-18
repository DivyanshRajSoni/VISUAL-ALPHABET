# from googletrans import Translator
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from flask import jsonify,Flask, request, render_template
import os
import time
import auth
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


app = Flask(__name__)


class TXT_TO_IMG_CFG:
    seed = 42
    if torch.cuda.is_available():
        device = "cuda"
        generator = torch.Generator(device).manual_seed(seed)
    else:
        device = "cpu"
        generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (500,500)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12


def get_model(hugging_token):
    image_gen_model = StableDiffusionPipeline.from_pretrained(
    TXT_TO_IMG_CFG.image_gen_model_id,
    revision="fp16", use_auth_token=hugging_token, guidance_scale=9)
    image_gen_model = image_gen_model.to(TXT_TO_IMG_CFG.device)
    return image_gen_model;


text_to_img_model = get_model(auth.auth_token);


def get_img_to_text_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model,feature_extractor,tokenizer,device


img_to_txt_model,feature_extractor,tokenizer,device = get_img_to_text_model();


class IMG_TO_TXT_CFG:
    max_length = 16
    num_beams = 4


def predict_step(image_paths):
    gen_kwargs = {"max_length": IMG_TO_TXT_CFG.max_length, "num_beams": IMG_TO_TXT_CFG.num_beams};
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = img_to_txt_model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds





# def get_translation(text,dest_lang):
#   translator = Translator()
#   translated_text = translator.translate(text, dest=dest_lang)
#   return translated_text.text




    
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=TXT_TO_IMG_CFG.image_gen_steps,
        generator=TXT_TO_IMG_CFG.generator,
        guidance_scale=TXT_TO_IMG_CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(TXT_TO_IMG_CFG.image_gen_size)
    return image
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')  # Get the prompt from the request
    hugging_token = request.form.get('auth_token')
    print(prompt)
    #translation = get_translation(prompt,'en')
    #print(translation)
    img = generate_image(prompt, text_to_img_model)
    # Save the image locally
    save_path = save_image(img)
    
    # Create a JSON response
    response = {
        'prompt': prompt,
        'image_path': save_path
    }
    
    return jsonify(response)

@app.route('/caption', methods=['POST'])
def captionIt():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_path = save_image(image_file)
    # image_path = "temp_image.jpg"  # Save the image temporarily
    # image_file.save(image_path)

    try:
        generated_text = predict_step([image_path])
        response = {"generated_text": generated_text[0]}  # Assuming you generate one caption per image
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

   
def generate_unique_filename():
    # Implement your logic to generate a unique filename
    # You can use a timestamp or any other approach
    
    # For example, generate a filename based on current timestamp
    timestamp = str(int(time.time()))
    filename = f"image_{timestamp}.jpg"
    
    return filename

def save_image(image):
    # Define the folder where the images will be saved
    save_folder = 'static'
    
    # Ensure the save folder exists, create it if necessary
    os.makedirs(save_folder, exist_ok=True)
    
    # Generate a unique filename for the image
    filename = generate_unique_filename()
    
    # Save the image to the folder
    image_path = os.path.join(save_folder, filename)
    image.save(image_path)
    print('image saved at this path', image_path)
    
    return image_path
    
    

   

#  for local
if __name__ == "__main__":
    app.run(debug=True)

#  for cloud
# if __name__ == "__main__":
#     app.run(host = '0.0.0.0',port=8080)