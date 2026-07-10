import gradio as gr
import torch
import os
import faiss
import numpy as np
import wandb
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env_setup import setup_env
setup_env()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize W&B only if API key is available
if os.environ.get("WANDB_API_KEY"):
    print("Initializing W&B...")
    wandb.login()
    wandb.init(project="DLA-Lab2-Retrieval", name="Gradio-Search-Logs", reinit=True)
else:
    print("⚠️  WANDB_API_KEY not set — W&B logging disabled. Add it to .env to enable.")

# 1. Load CLIP model and processor
model_id = "openai/clip-vit-base-patch16"
print("Loading CLIP model...")
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)
model.eval()

# 2. Load Dataset
print("Loading Flickr8k dataset...")
try:
    dataset = load_dataset("jxie/flickr8k", split="train")
except Exception as e:
    print(f"Could not load jxie/flickr8k ({e}). Falling back to nlphq/flickr8k...")
    dataset = load_dataset("nlphq/flickr8k", split="train")

subset_size = min(2000, len(dataset))
dataset = dataset.select(range(subset_size))

# 3. FAISS Indexing Mechanism
index_path = "flickr8k_faiss.index"  # Matches the jxie/flickr8k dataset used above

if os.path.exists(index_path):
    print("Loading pre-computed FAISS index...")
    index = faiss.read_index(index_path)
else:
    print("Computing image embeddings and building FAISS index for the first time...")
    image_embeddings = []
    
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            if "image" in batch:
                images = [img.convert("RGB") for img in batch["image"]]
            else:
                img_col = [c for c in dataset.column_names if c != "text" and c != "caption"][0]
                images = [img.convert("RGB") for img in batch[img_col]]
                
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = getattr(features, "image_embeds", getattr(features, "pooler_output", features))
            if isinstance(features, tuple):
                features = features[0]
                
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            image_embeddings.append(features.cpu().numpy().astype('float32'))
            
    embeddings_np = np.concatenate(image_embeddings, axis=0)
    
    print(f"Building FAISS Index (Inner Product for Cosine Similarity)...")
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    print(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)

# 4. Retrieval Function using FAISS
def retrieve_images(query, top_k=10):
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(text_features, "text_embeds", getattr(text_features, "pooler_output", text_features))
        if isinstance(text_features, tuple):
            text_features = text_features[0]
            
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features_np = text_features.cpu().numpy().astype('float32')
        
        # FAISS search
        similarities, top_indices = index.search(text_features_np, top_k)
        
        results = []
        for i, idx in enumerate(top_indices[0]):
            if "image" in dataset.column_names:
                img = dataset[int(idx)]["image"]
            else:
                img_col = [c for c in dataset.column_names if c != "text" and c != "caption"][0]
                img = dataset[int(idx)][img_col]
            results.append((img, f"Score: {similarities[0][i]:.3f}"))
            
        # --- W&B Interactive Logging ---
        if wandb.run is not None:
            # Log the query and the top 5 visual results to create a live search gallery
            wandb_images = [wandb.Image(img, caption=f"Rank {j+1} | {score}") for j, (img, score) in enumerate(results[:5])]
            wandb.log({
                "Query String": query,
                "Top 5 Retrieved Images": wandb_images
            })
            
        return results

# 5. Build Gradio UI
with gr.Blocks(title="Text-to-Image Retrieval (FAISS)", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ Ultra-Fast Text-to-Image Retrieval with CLIP & FAISS")
    gr.Markdown("Search through the Flickr8k dataset using natural language! Powered by `openai/clip-vit-base-patch16` and **FAISS** for instant vector search.")
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Search Query", 
                placeholder="e.g. a dog catching a frisbee in the park...",
                lines=1
            )
        with gr.Column(scale=1):
            search_button = gr.Button("Search", variant="primary")
            
    gallery = gr.Gallery(
        label="Top 10 Matching Images", 
        show_label=True, 
        elem_id="gallery",
        columns=[5], 
        rows=[2], 
        object_fit="contain", 
        height="auto"
    )
    
    search_button.click(fn=retrieve_images, inputs=query_input, outputs=gallery)
    query_input.submit(fn=retrieve_images, inputs=query_input, outputs=gallery)
    
    gr.Examples(
        examples=[
            "a dog catching a frisbee",
            "people playing baseball",
            "a beautiful sunset over the water",
            "children running in the snow"
        ],
        inputs=query_input
    )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    app.launch(server_name="0.0.0.0", share=True)
