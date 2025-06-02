r"""image processor"""

from transformers import (AutoModelForImageClassification,
                          AutoImageProcessor, ViTImageProcessor, ViTModel,AutoTokenizer, AutoModel)
import torch


class ImageProcessor:
    r"""image processor"""
    tag_model_name = 'p1atdev/wd-swinv2-tagger-v3-hf'

    vit_model_name = 'legekka/AI-Anime-Image-Detector-ViT'

    text_model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tag_model = None
        self._image_preprocess = None
        self._model = None
        self._feature_extractor = None
        self._text_model = None
        self._tokenizer=None

    @property
    def text_model(self):
        if self._text_model is None:
            self._text_model = AutoModel.from_pretrained(self.text_model_name).to(self.device)
        return self._text_model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer =  AutoTokenizer.from_pretrained(self.text_model_name)
        return self._tokenizer
        
    @property
    def tag_model(self):
        if self._tag_model is None:
            self._tag_model = AutoModelForImageClassification.from_pretrained(
                self.tag_model_name).to(self.device)
        return self._tag_model

    @property
    def image_preprocess(self):
        if self._image_preprocess is None:
            self._image_preprocess = AutoImageProcessor.from_pretrained(
                self.tag_model_name, trust_remote_code=True)
        return self._image_preprocess

    @property
    def model(self):
        if self._model is None:
            self._model = ViTModel.from_pretrained(
                self.vit_model_name).to(self.device)
        return self._model

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            self._feature_extractor = ViTImageProcessor.from_pretrained(
                self.vit_model_name)
        return self._feature_extractor

    def processing_feature(self, images):
        outputs = image_process(
            self.model, self.feature_extractor, images, self.device)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return [feature.tolist() for feature in features]

    def processing_tags(self, images,limit,threshold):
        outputs = image_process(
            self.tag_model, self.image_preprocess, images, self.device)
        result = [{self.tag_model.config.id2label[i]: logit.float().item() for i, logit in enumerate(torch.sigmoid(logits))} for logits in outputs.logits]
        result = [dict(sorted({k: v for k, v in res.items() if v > threshold}.items(), key=lambda item: item[1], reverse=True)) for res in result]
        return result[:limit]

    def text_embedings(self,docs,limit,threshold):
        inputs = self.tokenizer(docs, padding=True, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        with torch.no_grad():
            embeddings = self.text_model(**inputs).last_hidden_state[:, 0].cpu().detach().numpy()
            import numpy as np
            from numpy.linalg import norm
            target = embeddings[0]
            similarities = [np.dot(target,item)/(norm(target)*norm(item)) for item in embeddings[1:]]
            similarities = [{docs[i+1]:similarity.item()} for i,similarity in enumerate(similarities) if similarity > threshold]
            return similarities[:limit]

def image_process(use_model, preprocess, images, use_device):
    """
    Processes a list of images using the provided model and preprocessing function.

    Args:
        use_model (torch.nn.Module): The model to be used for processing the images.
        preprocess (callable): The preprocessing function to be applied to the images.
        images (list): A list of images to be processed.
        use_device (str): Device to use for processing ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The model outputs containing the processed image data.
    """
    inputs = preprocess(images=images, return_tensors='pt').to(use_device)
    with torch.no_grad():
        return use_model(**inputs)
