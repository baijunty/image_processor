from concurrent.futures import ThreadPoolExecutor

from fastai.vision.core import PILImage
from flask import Flask, jsonify, render_template, request, send_file
# from auto_color import AutoColor
from torchvision import transforms

from image_process import ImageProcessor
from translator import Translator
import torch
app = Flask(__name__)
translator = Translator()
image_processor = ImageProcessor()

# coloring = AutoColor()

to_tensor = transforms.ToTensor()
grayscale = transforms.Grayscale(num_output_channels=1)
resize = transforms.Resize((8, 8))
executor=ThreadPoolExecutor(max_workers=8)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/evaluate', methods=['POST'])
def upload_file():
    import time
    start = time.time()
    files = request.files.getlist("file")
    proces = request.values.get("process", '')
    threshold = float(request.values.get("threshold", '0.2'))
    limit = int(request.values.get("limit", '40'))
    # result = [{"fileName": file.filename,'tags':{label['label']:label['score'] for label in out if label['score'] > threshold }} for file,out in zip(files,pipe(images,top_k=limit))]
    result = []
    match proces:
        case 'tagger':
            if not files or len(files) == 0:
                return jsonify({'error': 'No file part'}), 400
            files = request.files.getlist("file")
            images = [PILImage.create(file) for file in files]
            result = image_processor.processing_tags(images, limit, threshold)
            for i, res in enumerate(result):
                result[i] = {"fileName": files[i].filename, "tags": dict(sorted({k: v for k, v in res.items(
                ) if v > threshold}.items(), key=lambda item: item[1], reverse=True)[:limit])}
        case 'feature':
            if files and len(files) > 0:
                images = [PILImage.create(file) for file in files]
                features = image_processor.processing_feature(images)
                result = [{"fileName": file.filename, 'data': feature}
                          for file, feature in zip(files, features)]
            elif request.values.get('docs'):
                import json
                data = request.values.get('docs')
                print('input ', data)
                docs = json.loads(data)
                result = image_processor.text_embedings(docs, limit, threshold)
            else:
                return jsonify({'error': 'No input part'}), 400
        case 'translate':
            if not files or len(files) == 0:
                return jsonify({'error': 'No file part'}), 400
            files = request.files.getlist("file")
            image = PILImage.create(files[0])
            import io
            import os

            import cv2
            import numpy
            lang = request.values.get("lang", 'ja')
            result = translator.translate(cv2.cvtColor(
                numpy.asarray(image), cv2.COLOR_RGB2BGR), src=lang)
            print(f'process translate {lang} time {time.time()-start}')
            return send_file(io.BytesIO(result), mimetype='image/jpg', as_attachment=True, download_name=f'{os.path.splitext(files[0].filename)[0]}.jpg')
        case 'image_hash':
            if not files or len(files) == 0:
                return jsonify({'error': 'No file part'}), 400
            files = request.files.getlist("file")
            futhres = [executor.submit(image_hash, PILImage.create(file)) for file in files]
            result = {}
            try:
                for i,fut in enumerate(futhres):
                    result[files[i].filename]=fut.result()
            except Exception as e:
                print(e)
            print('received ', [file.filename for file in files],'time ',time.time()-start)
            return jsonify(result)
    return jsonify(result)

@app.route('/resize', methods=['GET'])
def resize_image():
    """ 缩放图片 """
    file = request.args.get('file')
    import os,io
    from PIL import Image
    if not file or not os.path.exists(file):
        return jsonify({'error': 'File does not exist'}), 400
    image = Image.open(file).convert("RGB")
    width = request.args.get('width', type=int, default=384)
    height = request.args.get('height', type=int, default=width*image.height// image.width)
    resized_image = resize_image_to_target(image, width, height)
    output = io.BytesIO()
    resized_image.save(output, format='JPEG')
    output.seek(0)
    return send_file(output, mimetype='image/jpeg', as_attachment=True, download_name=f'{os.path.splitext(file)[0]}.jpg')

def resize_image_to_target(image, width, height):
    """
    使用PIL库的Image模块对图像进行缩放。

    Args:
        image (PIL.Image): 需要缩放的原始图像。
        width (int): 缩放后图像的目标宽度。
        height (int): 缩放后图像的目标高度。

    Returns:
        PIL.Image: 已经调整大小后的图像。

    Note:
        函数中的参数存在错误，请确保调用时传递正确的宽和高。
    """
    # 将图像转换为PyTorch张量
    tensor = to_tensor(image).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 应用调整大小变换，将张量从GPU转移到CPU
    tensor = transforms.Resize((height, width))(tensor).to('cpu')

    # 将张量转换回PIL图像
    image = transforms.ToPILImage()(tensor)

    return image

def hex_to_signed(hex_str, bits):
    """
    将十六进制字符串转换为有符号整数。

    Args:
        hex_str (str): 十六进制字符串。
        bits (int): 位数，用于确定有符号整数的范围。

    Returns:
        int: 转换后的有符号整数值。

    Note:
        如果输入的十六进制值超出指定位数的有符号整数范围，可能会返回错误的结果。
    """
    unsigned_val = int(hex_str, 16)
    mask = (1 << (bits - 1))
    if unsigned_val & mask:
        return unsigned_val - (1 << bits)
    else:
        return unsigned_val

def image_hash(image):
    from imagehash import ImageHash
    image = to_tensor(image).to('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = grayscale(image).to('cpu')
    tensor = resize(tensor)
    mean = tensor.mean()
    binary_hash = (tensor > mean).to(torch.uint8)
    pixels = binary_hash.squeeze(0).numpy()
    mean = pixels.mean()
    diff = pixels>mean
    hash=ImageHash(diff)
    return hex_to_signed(str(hash),64)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
