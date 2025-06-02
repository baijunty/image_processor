import math
import functools
from collections import Counter
# Import required packages
import cv2
import torch
from ultralytics import YOLO
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from easyocr import Reader
from craft_text_detector import load_craftnet_model,get_prediction

PRETRAINED_MODEL_NAME_OR_PATH = "kha-white/manga-ocr-base"


class Translator:

    request = """将以下文本从其原始语言翻译成"中文"。尽可能保持原文的语气和风格,不需要解释,只需要按原格式返回翻译后的json内容：
    文本：
    ---------
    {}
    ---------
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO('models/best.pt')
        self._processor = None
        self._tokenizer = None
        self._model = None
        # self.refine_net = load_refinenet_model(cuda=torch.cuda.is_available())
        self._craft_net = None
        self.reader = Reader(['en','ko'],gpu=self.device=='cuda')

    @property
    def processor(self):
        if self._processor is None:
            self._processor=ViTImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
        return self._processor

        
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer=AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
        return self._tokenizer

        
    @property
    def model(self):
        if self._model is None:
            self._model= VisionEncoderDecoderModel.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH).to(self.device)
        return self._model

    @property
    def craft_net(self):
        if self._craft_net is None:
            self._craft_net= load_craftnet_model(cuda=self.device=='cuda')
        return self._craft_net

    def craft_text_bubble_detect(self, image):
        prediction_result = get_prediction(image=image, craft_net=self.craft_net, refine_net=None,
                                           text_threshold=0.6, link_threshold=0.4, low_text=0.4, cuda=self.device=='cuda', long_size=max(image.shape),poly=False)
        result = []
        for box in prediction_result["boxes"]:
            # result_rgb = file_utils.rectify_poly(image,box)
            # result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            x, y = box[0]
            x1, y1 = box[2]
            result.append((math.floor(x.item()), math.floor(y.item()),
                           math.ceil(x1.item()), math.ceil(y1.item())))
        return result

    def __merge_conflict_area(self, boxes):
        new_boxes = []
        for box in boxes:
            conflict = False
            for i, item in enumerate(new_boxes):
                if self.__area_conflict(item, box) or self.__is_same_column(box, item):
                    new_boxes[i] = (min(box[0], item[0]), min(box[1], item[1]), max(
                        box[2], item[2]), max(box[3], item[3]))
                    conflict = True
                    break
            if not conflict:
                new_boxes.append(box)
        return new_boxes

    def jp_text_ocr(self, images):
        datas = self.processor(images, return_tensors="pt").pixel_values.squeeze()
        result = []
        for i, d in enumerate(datas):
            data = self.model.generate(
                d[None].to(self.device), max_length=300)[0]
            data = self.tokenizer.decode(data, skip_special_tokens=True)
            data = data.replace(' ', '')
            height, width = images[i][0].shape[:2]
            vert = height > width
            if vert:
                data = data.replace('\n', '')
            result.append(data)
        return result

    def __compute_text_size(self, draw, text_size):
        font = ImageFont.truetype(
            "NotoSansSC-Bold.ttf", text_size, encoding="utf-8")
        _, _, w, h = draw.textbbox((0, 0), '你', font=font)
        return font, w, h

    def __render_chinese_text(self, img, text, rect, color=(0, 0, 0)):
        if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        width, height = img.width, img.height
        text_size = 16
        ref_width = max(rect[2]-rect[0], width*0.7)
        ref_height = max(rect[3]-rect[1], height*0.7)
        top = (height-ref_height)/2
        font, w, h = self.__compute_text_size(draw, text_size)
        words = self.__split_str(text, h, ref_height)
        deep = max([len(word) for word in words])
        while w*len(words)*1.2 <ref_width and text_size<35:
            text_size += 1
            font, w, h = self.__compute_text_size(draw, text_size)
            words = self.__split_str(text, h, ref_height)
            deep = max([len(word) for word in words])
        # print(f'{text} {rect} use w {w} h {h} size {text_size} {deep} preferer {top} len {len(words)} rende to {ref_width}:{ref_height} color {color}')
        for i in range(deep):
            left=(width-len(words)*w)/2
            if len(words) > 1:
                for word in reversed(words):
                    if len(word) > i:
                        _, _, rw, _ = draw.textbbox((0, 0), word[i], font=font)
                        draw.text((left+(w-rw), top), word[i],spacing=0, fill=color, font=font, language='zh-Hans')
                    left+=w
            else:
                draw.text((left, top),
                          text[i], color, font=font, language='zh-Hans')
            top += h
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def __split_str(self, text, h, heigh):
        # import re
        # is_punct=False
        words = []
        pov = 0
        for i, _ in enumerate(text):
            # is_word=re.fullmatch(r"[\u4e00-\u9fa5|\w]",t) is None
            if (i-pov+1)*h >= heigh:
                words.append(text[pov:i])
                pov = i
            # is_punct= is_word
        words.append(text[pov:])
        return words

    def translate_text(self, data, src='ja', dest='zh'):
        import requests
        import json
        resp= requests.get(f'https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl={src}&tl={dest}&q={data}', timeout=50).json()[0]
        # print(f'{data} return {resp}')
        return resp[0][0].replace('…', '.').replace('“', '"').replace('”', '"').replace('、', ',').replace('【', '[').replace('】', ']') if resp else ''
        # result = requests.get(f'https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl=ja&tl=zh&q={json.dumps(data, ensure_ascii=False)}', timeout=50).json()[
        #     0][0][0].replace('…', '.').replace('“', '"').replace('”', '"').replace('、', ',').replace('【', '[').replace('】', ']')
        # print(f'input {json.dumps(data, ensure_ascii=False)} return {result}')
        # return json.loads(result)
        # resp=  requests.post('http://127.0.0.1:11434/api/generate',data=json.dumps({"model": "qwen2.5","prompt": self.request.format(json.dumps(data)),"stream": False,'system':"你是一个翻译专家,根据用户输入内容翻译文本"}),timeout=50).json()
        # resp=resp['response'].replace('…','..').replace('```','').replace('\'','\"')
        # print(f'input {data} return type {type(resp)} data {resp}')
        # translated=json.loads(resp)
        # return translated

    def __text_area_compare(self, b, b1):
        result = 0
        if self.__is_same_line(b, b1):
            result = -1 if b[0] > b1[0] else 1
        else:
            result = b[1]-b1[1]
        return result

    def __is_same_line(self, ele, box):
        length = min(ele[3]-ele[1], box[3]-box[1])
        return (box[1] >= ele[1] and box[1] <= ele[3] and (ele[3]-box[1]) > length*0.4) or (ele[1] >= box[1] and ele[1] <= box[3] and (box[3]-ele[1]) > length*0.4)

    def __is_same_column(self, ele, box):
        length = min(ele[2]-ele[0], box[2]-box[0])
        return (box[0] >= ele[0] and box[0] <= ele[2] and (ele[2]-box[0]) > length*0.8) or (ele[0] >= box[0] and ele[0] <= box[2] and (box[2]-ele[0]) > length*0.8)

    def __area_conflict(self, b1, b2):
        return not (b1[2] < b2[0] or b1[3] < b2[1] or b1[0] > b2[2] or b1[1] > b2[3])

    def __sort_areaes(self, boxes):
        result = []
        while len(boxes) > 0:
            box = boxes.pop()
            exist = False
            for element in result:
                for ele in element:
                    if self.__is_same_line(ele, box):
                        exist = True
                        break
                if exist:
                    element.append(box)
                    break
            if not exist:
                result.append([box])
        sort_box = []
        result.sort(key=lambda l: l[0][1])
        from functools import cmp_to_key
        for r in result:
            re = sorted(r, key=cmp_to_key(mycmp=self.__text_area_compare))
            sort_box.extend(re)
        return sort_box

    def __check_orderly_rects(self, boxes, rect):
        mid = (rect[2]-rect[0])/2
        length = min(boxes[-1][2]-mid, mid-boxes[0][0])
        space = length
        item = boxes[0]
        for i, box in enumerate(boxes[1:]):
            gap = box[0]-item[2] if box[0] > item[2] else 0
            space = min(gap, space) if gap > 0 else space
            item = box
        space = max(space, 8)
        # print(f'{rect} has {boxes} mid {mid} min len {length} space {space}')
        result = []
        for i, box in enumerate(boxes):
            left = boxes[i-1] if i > 0 else None
            right = boxes[i+1] if i < len(boxes)-1 else None
            if left and abs(box[0]-left[2]) < space*3 and min(abs(mid-box[0]), abs(box[2]-mid)) < length*1.36:
                result.append(box)
            elif right and abs(box[2]-right[0]) < space*3 and min(abs(mid-box[0]), abs(box[2]-mid)) < length*1.36:
                result.append(box)
        return result

    def find_text_area(self, image, rect,color):
        # import time
        boxes = self.craft_text_bubble_detect(image)
        boxes.sort(key=lambda l: l[0])
        boxes = self.__merge_conflict_area(boxes)
        if len(boxes) == 0:
            print('rect detetor fail ', rect)
            return None
        elif len(boxes) == 1:
            box = boxes[0]
            if color:
                text_img=image[box[1]:box[3],box[0]:box[2]]
                text_img[:] = color
        else:
            rects = self.__check_orderly_rects(boxes, rect)
            if len(rects) == 0:
                return None
            if color:
                for box in rects:
                    text_img=image[box[1]:box[3],box[0]:box[2]]
                    text_img[:] = color
            box = functools.reduce(lambda r1, r2: (min(r1[0], r2[0]), min(r1[1], r2[1]), max(
                r1[2], r2[2]), max(r1[3], r2[3])), rects) if len(rects) > 1 else rects[0]
        return box

    def common_text_ocr(self, images,rects):
        result = []
        for i,(image,color) in enumerate(images):
            r = self.reader.readtext(image)
            if len(r) == 0:
                result.append(('',None))
                continue
            content = ''
            boxes=[]
            box=self.find_text_area(image, rects[i],None)
            if box:
                boxes.append(box)
            for point, text, conf in r:
                content += text.lower()
                box=(math.floor(point[0][0]), math.floor(point[0][1]), math.ceil(point[2][0]), math.ceil(point[2][1]))
                text_img=image[box[1]:box[3],box[0]:box[2]]
                text_img[:] = color
                boxes.append(box)
            box = functools.reduce(lambda r1, r2: (min(r1[0], r2[0]), min(r1[1], r2[1]), max(
                r1[2], r2[2]), max(r1[3], r2[3])), boxes)
            result.append((content,box))
        return result

    def translate(self, image: str | np.ndarray, src: str = 'ja', dest: str = 'zh'):
        if isinstance(image, str):
            image = cv2.imread(image)
        import time
        img = image
        start=time.time()
        boxes = self.yolo.predict(img)[0]
        if len(boxes)==0:
            return cv2.imencode(".jpg", img)[1].tobytes()
        boxes = [[round(x.item()) for x in cnt.boxes[0].xyxy[0]] for cnt in boxes]
        boxes = self.__sort_areaes(boxes)
        print(f'boxes {boxes}')
        if len(boxes)==0:
            return cv2.imencode(".jpg", img)[1].tobytes()
        print(f'yolo detetor time {time.time()-start} len {len(boxes)}')
        start=time.time()
        # for box in boxes:
        #     cv2.rectangle(img, (box[0], box[1]),
        #                   (box[2], box[3]), (0, 255, 0), 2)
        bubble_images=[]
        for cnt in boxes:
            croped=img[cnt[1]:cnt[3], cnt[0]:cnt[2]]
            cimg = Image.fromarray(cv2.cvtColor(croped, cv2.COLOR_BGR2RGB))
            cimg.thumbnail((64,64))
            counter = Counter(list(cimg.getdata())).most_common(1)
            color = counter[0][0]
            bubble_images.append((croped,color))
        print(f'count color time {time.time()-start} len {len(bubble_images)}')
        start=time.time()
        if src == 'ja':
            origin_text = self.jp_text_ocr([image for (image,color) in bubble_images])
            print(f'ocr detetor time {time.time()-start}')
            start=time.time()
            origin_text = [self.translate_text(text, src=src, dest=dest) for text in origin_text]
            text_area = [(text,self.find_text_area(croped, box,color)) for text, box, (croped,color) in zip(origin_text, boxes, bubble_images)]
        else:
            origin_text = self.common_text_ocr(bubble_images,boxes)
            print(f'ocr detetor time {time.time()-start}')
            start=time.time()
            text_area = [(self.translate_text(text, src=src, dest=dest),box) if len(text)>0 else (text,box) for (text,box) in origin_text]
        print(f'translate time {time.time()-start}')
        start=time.time()
        for cnt, text, box in [[cnt, text, box] for cnt, (text, box) in zip(boxes,text_area) if box and len(text)>0]:
            x, y, x1, y1 = cnt
            croped = img[y:y1, x:x1]
            text_img = croped[math.floor(box[1]):math.ceil(
                box[3]), math.floor(box[0]):math.ceil(box[2])]
            text_img = self.__render_chinese_text(
                croped, text, box, color=tuple(255-x for x in color))
            img[y:y1, x:x1] = text_img
        print(f'render time {time.time()-start}')
        return cv2.imencode(".jpg", img)[1].tobytes()
