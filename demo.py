import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("ViT-B-32.pt", device=device)# 本地的pt模型的路径

image = preprocess(Image.open("dog.png")).unsqueeze(0).to(device) # output score: [[0.9927805  0.00721956]]
text = clip.tokenize(["a cat", "a picture"]).to(device)# output score: [[0.40114564 0.59885436]]
#text = clip.tokenize(["a dog", "a picture"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features = image_features / image_features.norm(dim=1, keepdim=True) # 归一化后的图像特征向量
    text_features = text_features / text_features.norm(dim=1, keepdim=True) # 归一化后的文本特征向量

    logit_scale = model.logit_scale.exp()# 尺度缩放因子，就是固定的值: 100
    logits_per_image = logit_scale * image_features @ text_features.t()
    #logits_per_image = image_features @ text_features.t() # 这里得到的就是余弦相似度都是0～1的值，但是因为这个版本的clip的训练问题，相似度的绝对值都不是很大，需要对比才能凸显出来具体哪一个图文对的相似度更高，因此添加了一个没用的文本， “a picture”, 当图片和文本的相似度高的时候，自然比图片和“a picture”之间的相似度高，当图片和文本的相似度低的时候，大概率比图片和“a picture”之间的相似度低
    # 问题详见https://stackoverflow.com/questions/77732247/clip-cosine-similarity-of-text-and-image-embeddings-is-low
    # 问题暂时还没有解决
    logits_per_text = logits_per_image.t()

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", probs)

    final_prob = probs[0][0] # 假设只有一张图和一个文本(“a picture”之外)计算相似度
    print(final_prob)
    if final_prob > 0.5:
        print("real news")
    else:
        print("fake news")

