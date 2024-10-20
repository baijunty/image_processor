
import json
import os
import urllib.parse

import cv2
import numpy as np
import requests
from ultralytics import YOLO
from websockets.sync.client import connect

from image_process import ImageProcessor


class AutoColor:
    image_process = ImageProcessor()
    remove_keys = ["monochrome", "greyscale", 'background',"comic",'speech bubble']
    comfyui_addr = "192.168.1.242:8188"

    def __init__(self):
        self._model = None
        import uuid
        self.client_id = uuid.uuid4().hex
        self._ws=None
        self._mask_model=None
        with open("ai_write.json", "r", encoding='utf-8') as f:
            self.workflow = json.load(f)

    @property
    def mask_model(self):
        if self._mask_model is None:
            self._mask_model=YOLO("person_yolov8.pt")
        return self._mask_model

    @property
    def ws(self):
        if self._ws is None:
            self._ws = connect(f"ws://{self.comfyui_addr}/ws?clientId={self.client_id}")
        return self._ws

    @property
    def model(self):
        if self._model is None:
            self._model = YOLO("person.pt")
        return self._model


    def convert_to_transparent(self,image):
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=image.dtype) * 255
        return cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel])


    def coloring_by_stable_diffusion(self, image, prompt, name):
        height, width = image.shape[:2]
        # background = np.zeros(image.shape, dtype=np.uint8)
        # background.fill(255)
        # image = self.merge_image(image,background)
        url = f"http://{self.comfyui_addr}/upload/image"
        data = {'subfolder': 'autocolor',"overwrite":True}
        response = requests.post(url, files={'image': (name, cv2.imencode(".png", image)[1].tobytes(), 'image/png')}, data=data, timeout=60)
        print(f'response {response}')
        if response.status_code == 200:
            resp = response.json()
            self.workflow['client_id'] = self.client_id
            self.workflow['prompt']['252']['inputs']['text'] = prompt
            import random
            self.workflow['prompt']['19']['inputs']['seed'] = random.randint(0, 9029780681674491)
            self.workflow['prompt']['111']['inputs']['image'] = os.path.join(resp['subfolder'], resp['name'])
            print(f'{name} prompt {prompt} resp {resp}')
            response = requests.post(f"http://{self.comfyui_addr}/prompt", data=json.dumps(self.workflow).encode('utf-8'), timeout=60).json()
            print(f'response {response}')
            prompt_id = response['prompt_id']
            try:
                color_img=self.get_images(prompt_id)
                color_img= cv2.resize(color_img,(width,height))
                return  self.convert_to_transparent(color_img) if color_img.shape[2] == 3 else color_img
            except Exception as e:
                print(e)
        return self.convert_to_transparent(image) if image.shape[2] == 3 else image

    def get_transparent_image(self, image):
        # from PIL import Image
        # mask_, alpha_image = segment_rgba_with_isnetis(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        # mask_inv = cv2.bitwise_not(mask_)
        # background = np.dstack([image, mask_inv])
        # return cv2.cvtColor(np.asarray(alpha_image), cv2.COLOR_RGBA2BGRA),background
        result = self.mask_model.predict(image)[0]
        b_mask = np.zeros(image.shape[:2], np.uint8)
        # white_image = np.zeros((*image.shape[:2],4), np.uint8)
        # white_image.fill(255)
        for cnt in result:
            box = [round(x.item()) for x in cnt.boxes[0].xyxy[0]]
            contour = cnt.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            alpha_image = np.dstack([image, b_mask])
            mask_inv = cv2.bitwise_not(b_mask)
            background = np.dstack([image, mask_inv])
            return alpha_image,background
    
    def merge_image(self, image, background):
        b1, g1, r1, a1 = cv2.split(image)
        b2, g2, r2, a2 = cv2.split(background)
        a1 = a1 / 255.0
        a2 = a2 / 255.0
        b = b1 * a1 + b2 * a2 * (1 - a1)
        g = g1 * a1 + g2 * a2 * (1 - a1)
        r = r1 * a1 + r2 * a2 * (1 - a1)
        a = (a1 + a2 * (1 - a1)) * 255
        a = np.clip(a, 0, 255).astype(np.uint8)
        return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8), a])
    
    
    def coloring(self, image, name):
        if isinstance(image, str):
            image = cv2.imread(image)
        height, width = image.shape[:2]
        print(f'image shape {image.shape}')
        result = self.model.predict(image)[0]
        return_image = self.convert_to_transparent(image)
        for i, cnt in enumerate(result):
            box = [round(x.item()) for x in cnt.boxes[0].xyxy[0]]
            # isolated = np.dstack([image, b_mask])
            # croped = isolated[box[1]:box[3], box[0]:box[2]]
            origin_image = image[box[1]:box[3], box[0]:box[2]]
            height, width = origin_image.shape[:2]
            print(f"Image size: {width}x{height}")
            if width * height < 128*256 :
                continue
            elif width * height > 960*1280:
                croped = cv2.resize(origin_image, (960, round(height*960/width))) if height>width else cv2.resize(origin_image, (round(width*960/height), 960))
            else:
                croped = cv2.resize(origin_image, (640, round(height*640/width))) if height>width else cv2.resize(origin_image, (round(width*640/height), 640))
            prompt=''
            for tag in self.image_process.processing_tags(croped, 20, 0.5)[0].keys():
                illegal=list(filter(lambda x: tag.find(x)!=-1, self.remove_keys))
                if len(illegal)==0:
                    prompt += f'{tag.split(':').pop()},'
            alpha_image,background = self.get_transparent_image(croped)
            if alpha_image is None:
                return_image[box[1]:box[3], box[0]:box[2]] = origin_image
                continue
            color_img = self.coloring_by_stable_diffusion(alpha_image, prompt, f'{name}_{i}.png')
            merged_img=self.merge_image(color_img, background)
            color_img= cv2.resize(merged_img, (width, height))
            return_image[box[1]:box[3], box[0]:box[2]] = color_img
            # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            print(f'detetor {box} conf {cnt.boxes[0].conf.item()}')
        return cv2.imencode(".png", return_image)[1].tobytes()

    def queue_prompt(self,prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = requests.get(f"http://{self.comfyui_addr}/prompt", data=data,timeout=60).json()
        print(f'response {req}')
        return req
        
    def get_image(self,filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        return requests.get(f"http://{self.comfyui_addr}/view?{url_values}",timeout=60).content
        
    def get_history(self,prompt_id):
        return requests.get(f"http://{self.comfyui_addr}/history/{prompt_id}",timeout=60).json()
        
    def get_images(self, prompt_id):
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    return cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return None