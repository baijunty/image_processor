---
library_name: transformers
license: apache-2.0
base_model: SmilingWolf/wd-swinv2-tagger-v3
inference: false
tags:
- wd-tagger
- optimum
---

# WD SwinV2 Tagger v3 with 🤗 transformers

Converted from [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) to transformers library format.

## Example

[![](https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/gist/p1atdev/d420d9fcd5c8ea66d9e10918fc330741/wd-swinv2-tagger-v3-hf-pipe.ipynb)

### Installation

```bash
pip install transformers
```

### Pipeline

```py
from transformers import pipeline

pipe = pipeline(
    "image-classification",
    model="p1atdev/wd-swinv2-tagger-v3-hf",
    trust_remote_code=True,
)

print(pipe("sample.webp", top_k=15))
#[{'label': '1girl', 'score': 0.9973934888839722},
# {'label': 'solo', 'score': 0.9719744324684143},
# {'label': 'dress', 'score': 0.9539461135864258},
# {'label': 'hat', 'score': 0.9511678218841553},
# {'label': 'outdoors', 'score': 0.9438753128051758},
# ...
```


### AutoModel


```py
from PIL import Image

import numpy as np
import torch

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

MODEL_NAME = "p1atdev/wd-swinv2-tagger-v3-hf"

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

image = Image.open("sample.webp")
inputs = processor.preprocess(image, return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs.to(model.device, model.dtype))
logits = torch.sigmoid(outputs.logits[0]) # take the first logits

# get probabilities
results = {model.config.id2label[i]: logit.float() for i, logit in enumerate(logits)}
results = {
    k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True) if v > 0.35 # 35% threshold
}
print(results)  # rating tags and character tags are also included
#{'1girl': tensor(0.9974),
# 'solo': tensor(0.9720),
# 'dress': tensor(0.9539),
# 'hat': tensor(0.9512),
# 'outdoors': tensor(0.9439),
# ...
```

### Accelerate with 🤗 Optimum

Maybe about 30% faster and about 50% light weight model size than transformers version, but the accuracy is slightly degraded.

```bash
pip install optimum[onnxruntime] 
```

```diff
-from transformers import pipeline
+from optimum.pipelines import pipeline

pipe = pipeline(
    "image-classification",
    model="p1atdev/wd-swinv2-tagger-v3-hf",
    trust_remote_code=True,
)

print(pipe("sample.webp", top_k=15))
#[{'label': '1girl', 'score': 0.9966088533401489},
# {'label': 'solo', 'score': 0.9740601778030396},
# {'label': 'dress', 'score': 0.9618403911590576},
# {'label': 'hat', 'score': 0.9563733339309692},
# {'label': 'outdoors', 'score': 0.945336639881134},
# ...
```


## Labels

All of rating tags have prefix `rating:` and character tags have prefix `character:`.

- Rating tags: `rating:general`, `rating:sensitive`, ...
- Character tags: `character:frieren`, `character:hatsune miku`, ...