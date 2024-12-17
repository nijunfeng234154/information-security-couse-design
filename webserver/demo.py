from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import StableDiffusionPipeline, DDIMScheduler
from flask_socketio import SocketIO, emit
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import make_response, jsonify
from PIL import Image, ImageOps
from loguru import logger
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
import numpy as np
import argparse
import binascii
import base64
import torch
import cv2
import os


#配置跨域请求
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False
GUIDANCE_SCALE = 1.0
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f"Using device: {device}")
ldm_stable = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=MY_TOKEN,
                                                     scheduler=scheduler).to(device)


try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


def load_image(image_path, left=0, right=0, top=0, bottom=0, resize=False):
    image = Image.open(image_path).convert("RGB")
    h, w = image.size
    if resize:
        image = np.array(image.resize((512, 512)))
    else:
        width_padding = -h % 8
        height_padding = -w % 8

        padded_image = ImageOps.expand(image, (
            0, 0, width_padding, height_padding),
                                       fill=None,
                                       )
        image = np.array(padded_image)
    return image, h, w

class ODESolve:

    def __init__(self, model, NUM_DDIM_STEPS=50):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        if context is None:
            context = self.context
        guidance_scale = GUIDANCE_SCALE
        uncond_embeddings, cond_embeddings = context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str):
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        return text_embeddings

    @torch.no_grad()
    def ddim_loop(self, latent, is_forward=True):
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps)):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)
            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def save_inter(self, latent, img_name):
        image = self.latent2image(latent)
        cv2.imwrite(img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def invert(self, prompt, start_latent, is_forward):
        self.init_prompt(prompt)
        latents = self.ddim_loop(start_latent, is_forward=is_forward)
        return latents[-1]

    def invert_i2n2i(self, prompt1, prompt2, image_start_latent, flip=False):

        self.init_prompt(prompt1)
        latent_i2n = self.ddim_loop(image_start_latent, is_forward=True)
        xT = latent_i2n[-1]

        if flip:
            xT = torch.flip(xT, dims=[2])

        self.init_prompt(prompt2)
        latent_n2i = self.ddim_loop(xT, is_forward=False)

        return self.latent2image(image_start_latent), image_start_latent, self.latent2image(latent_n2i[-1]), latent_n2i[
            -1]

# 计算均方误差 (MSE)
def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# 计算峰值信噪比 (PSNR)
def psnr(image1, image2):
    # 首先计算均方误差
    mse_value = mse(image1, image2)
    
    # 如果均方误差为零，则两图完全相同，PSNR为无限大
    if mse_value == 0:
        return float('inf')
    
    # 图像的最大像素值
    max_pixel = 255.0
    
    # 计算PSNR
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse_value)
    return psnr_value


ode = ODESolve(ldm_stable, 50)


def AuxReply(code, msg, data, fcode=200):
    return jsonify({"code": code, "msg": msg, "data": data}), fcode


@app.route('/logs', methods=['GET'])
def get_logs():
    with open('app.log', 'r') as f:
        logs = f.readlines()
    return jsonify(logs)

@logger.catch()
@app.route('/encrypt', methods=['POST'])
def encrypt():
    logger.info("接收到隐藏请求")
    from random import randint
    from datetime import datetime
    save_path = Path("./save/")

    if not save_path.exists() or not save_path.is_dir():
        logger.info(f"文件夹未创建 {save_path} 准备创建")
        save_path.mkdir()

    logger.info(f"开始解析请求体")
    data = request.get_json(force=True)  # 确保可以解析JSON
    origin_image_base64 = data.get("image")
    prompt_1 = data.get("key")
    prompt_2 = data.get("pub")
    decrypt_flag = bool(data.get("decrypt", False))

    logger.info(f"JSON参数解析完毕，开始转换图片，是否为揭示模式: {decrypt_flag}")

    # 去掉 Base64 编码中的前缀部分
    if origin_image_base64.startswith('data:image'):
        origin_image_base64 = origin_image_base64.split(';base64,').pop()

    # 确保 Base64 长度是 4 的倍数，如果不是，则添加适当的填充
    padding = '=' * (-len(origin_image_base64) % 4)
    origin_image_base64 += padding

    try:
        origin_image = Image.open(BytesIO(base64.b64decode(origin_image_base64)))
    except binascii.Error as e:
        logger.error(f"Base64解码失败: {e}")
        return jsonify({"status": "error", "message": "无效的Base64编码"}), 400
    
#{datetime.now().strftime('%Y%m%d_%H%M%S')}_{randint(10000, 99999):05}_
    origin_save = save_path / f"{'de_origin' if decrypt_flag else 'en_origin'}.png"
    origin_image.save(str(origin_save.resolve()))
    assert origin_save.exists() and origin_save.is_file()

    if decrypt_flag:
        decrypt_save = str(origin_save.resolve()).replace("_origin", "_decryptDone")
        return decrypt(origin_image_base64, prompt_2, prompt_1, decrypt_save)

    offsets = (0, 0, 0, 0)
    image_gt, h, w = load_image(origin_save, *offsets, resize=True)
    image_gt_latent = ode.image2latent(image_gt)
    result1 = cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR)
    latent_img_path = str(origin_save.resolve()).replace("_origin.png", "_latent.png")
    cv2.imwrite(latent_img_path, result1)

    # hide process
    latent_noise = ode.invert(prompt_1, image_gt_latent, is_forward=True)
    image_hide_latent = ode.invert(prompt_2, latent_noise, is_forward=False)

    # save container image
    image_hide = ode.latent2image(image_hide_latent)
    container = cv2.cvtColor(image_hide, cv2.COLOR_RGB2BGR)
    container_path = latent_img_path.replace("_latent.png", "_hide.png")
    cv2.imwrite(container_path, container)

    _, buffer = cv2.imencode('.png', container)
    container_base64 = base64.b64encode(buffer).decode('utf-8')
    
    response = make_response(jsonify({"status": "success", "data": container_base64}), 200)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    # return AuxReply(200, "success", container_base64)


#处理加噪声
def decode_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def decrypt(image_hide_base64, pub, key, save_file: str):
    logger.info(f"开始揭示，检查密语 Pub:{pub} Key:{key}")
    image_hide = Image.open(BytesIO(base64.b64decode(image_hide_base64)))
    image_hide = np.array(image_hide)  # Convert PngImageFile to np.ndarray
    rev_prompt_1 = key
    rev_prompt_2 = pub
    image_hide_latent_reveal = ode.image2latent(image_hide)
    latent_noise = ode.invert(rev_prompt_2, image_hide_latent_reveal, is_forward=True)
    image_reverse_latent = ode.invert(rev_prompt_1, latent_noise, is_forward=False)
    image_reverse = ode.latent2image(image_reverse_latent)
    image_reverse_rgb = cv2.cvtColor(image_reverse, cv2.COLOR_BGR2RGB)
    logger.success(f"揭示完成")
    _, buffer = cv2.imencode('.png', image_reverse_rgb)
    image_reverse_base64 = base64.b64encode(buffer).decode('utf-8')
    image_reverse_pil = Image.fromarray(image_reverse_rgb)
    image_reverse_pil.save(save_file)
    save_path = Path("./save/")
    # image_reverse = decode_image(image_reverse_base64)
    # image_reverse.save(save_path / f'decode_origin.png')
    
    #求原始图像和恢复出的图像的峰值信噪比

    origin_image = cv2.imread(save_path / 'en_origin.png')
    image_resolve = cv2.imread(save_file)
    
    image1 = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image_resolve, cv2.COLOR_BGR2GRAY)
    
    psnr_value = psnr(image1, image2)

    #可能也需要加header
    response = make_response(jsonify({"status": "success", "data": image_reverse_base64, "psnr":psnr_value}), 200)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    # return AuxReply(200, "success", image_reverse_base64)

def add_noise(img, choice):
    if choice == '1':
        # Compression Transform: Increase JPEG quality to reduce artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Increased from 50 to 80
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
    elif choice == '2':
        # Reduced Random Cropping: Crop less of the image
        h, w, _ = img.shape
        crop_ratio = 0.1  # Crop 10% from each side
        top = int(h * crop_ratio)
        bottom = int(h * (1 - crop_ratio))
        left = int(w * crop_ratio)
        right = int(w * (1 - crop_ratio))
        img = img[top:bottom, left:right]
    elif choice == '3':
        # Reduced Noise Intensity
        if np.random.rand() > 0.5:
            # Gaussian Noise: Reduce sigma to decrease intensity
            mean = 0
            sigma = 10  # Reduced from 25 to 10
            gauss = np.random.normal(mean, sigma, img.shape).astype('uint8')
            img = cv2.add(img, gauss)
        else:
            # Salt-and-Pepper Noise: Reduce amount to decrease intensity
            s_vs_p = 0.5
            amount = 0.01  # Reduced from 0.04 to 0.01
            out = np.copy(img)
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            out[tuple(coords)] = 255  # Corrected from 1 to 255 for proper salt effect

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            out[tuple(coords)] = 0
            img = out
    elif choice == '4':
        # Lighting Changes: Reduce range to make adjustments less drastic
        value = np.random.randint(-20,20)  # Reduced from -50 to 50
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    img_base64 = encode_image(img)
    return img_base64

@app.route('/noise', methods=['POST'])
def noise():
    logger.info("Received noise request")
    data = request.get_json(force=True)
    origin_image_base64 = data.get("image")
    choice = data.get("choice")

    if origin_image_base64 is None or choice is None:
        return jsonify({"error": "Invalid input"}), 400

    img = decode_image(origin_image_base64)
    img_with_noise_base64 = add_noise(img, choice)

    response = make_response(jsonify({"status": "success", "data": img_with_noise_base64}), 200)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

    # return jsonify({"image": result_image_base64})

# def find_fsr(image):
#     scale_factor = 2
#     height, width = image.shape[:2]
#     new_dimensions = (width * scale_factor, height * scale_factor)
#     fsr_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
#     return fsr_image


    

    # response = make_response(jsonify({"status": "success", "data": img_with_noise_base64}), 200)
    # response.headers.add('Access-Control-Allow-Origin', '*')
    # return response

def demo(params):
    if not os.path.exists(params.save_path):
        os.makedirs(params.save_path)

    ode = ODESolve(ldm_stable, params.num_steps)

    image_path = params.image_path
    prompt_1 = params.private_key
    prompt_2 = params.public_key

    rev_prompt_1 = prompt_1
    rev_prompt_2 = prompt_2
    need_flip = False

    offsets = (0, 0, 0, 0)
    image_gt, h, w = load_image(image_path, *offsets, resize=True)

    image_gt_latent = ode.image2latent(image_gt)
    cv2.imwrite("{:s}/gt.png".format(params.save_path), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

    # hide process
    latent_noise = ode.invert(prompt_1, image_gt_latent, is_forward=True)
    image_hide_latent = ode.invert(prompt_2, latent_noise, is_forward=False)

    # save container image
    image_hide = ode.latent2image(image_hide_latent)
    cv2.imwrite("{:s}/hide.png".format(params.save_path), cv2.cvtColor(image_hide, cv2.COLOR_RGB2BGR))

    # reveal process
    image_hide_latent_reveal = ode.image2latent(image_hide)
    latent_noise = ode.invert(rev_prompt_2, image_hide_latent_reveal, is_forward=True)

    image_reverse_latent = ode.invert(rev_prompt_1, latent_noise, is_forward=False)
    image_reverse = ode.latent2image(image_reverse_latent)
    cv2.imwrite("{:s}/reverse.png".format(params.save_path), image_reverse)

# 添加SocketIOHandler到logger
# logger.add(SocketIOHandler(), level='INFO')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    # logger.add("app.log", level='INFO')
    
    # socketio.run(app, debug=True)
    # demo(args)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', type=str, default='./asserts/1.png', help='test image path')
    # parser.add_argument('--private_key', type=str, default='Effiel tower', help='text prompt of the private key')
    # parser.add_argument('--public_key', type=str, default='a tree', help='text prompt of the public key')
    # parser.add_argument('--save_path', type=str, default='./output', help='text prompt of the public key')
    # parser.add_argument('--num_steps', type=int, default=50, help='sampling step of DDIM')
    # parser.add_argument('--guidance_scale', type=float, default=1.0, help='guidance scale for conditional generation')
    # parser.add_argument('--beta_start', type=float, default=0.00085, help='beta start for DDIM scheduler')
    # parser.add_argument('--beta_end', type=float, default=0.012, help='beta end for DDIM scheduler')
    # parser.add_argument('--beta_schedule', type=str, default="scaled_linear", help='beta schedule for DDIM scheduler')
