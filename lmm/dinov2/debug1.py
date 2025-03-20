import requests
from PIL import Image

from transformers import AutoImageProcessor

from .modeling_dinov2 import Dinov2Model
from .utils import visualize_feature

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = Dinov2Model.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

embeddings = outputs.last_hidden_state
cls_, fea_ = embeddings[:, 0], embeddings[:, 1:].reshape(1, 16, 16, -1)

# visualize_feature_l2(fea_.detach(), size=(16, 16), filename="mask_l2.png")
visualize_feature(fea_.detach(), size=(16, 16), filename="mask.png")

# # Reshape the feature tensor for PCA
# features = fea_.detach().reshape(-1, fea_.shape[-1]).numpy()

# # Perform PCA to reduce dimensions to 3
# pca = PCA(n_components=3)
# pca_features = pca.fit_transform(features)

# # Normalize the PCA components to [0, 1] for visualization
# pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

# # Reshape back to the original spatial dimensions
# pca_features = pca_features.reshape(fea_.shape[1], fea_.shape[2], 3)

# # Plot the PCA components as an image
# plt.figure(figsize=(8, 8))
# plt.imshow(pca_features)
# plt.axis('off')
# plt.title("Feature Visualization")

# # Save the figure
# plt.savefig("mask1.png", bbox_inches='tight', pad_inches=0.1)
# plt.close()  # Close the figure to free up memory