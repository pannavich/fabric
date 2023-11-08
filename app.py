import functools

import gradio as gr
import torch

from fabric.generator import AttentionBasedGenerator


#model_name = "dreamlike-art/dreamlike-photoreal-2.0"
model_name = "SG161222/Realistic_Vision_V2.0"
model_ckpt = "https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaper_7_pruned.safetensors"
model_ckpt = ""

class GeneratorWrapper:
  def __init__(self, model_name=None, model_ckpt=None):
    self.model_name = model_name if model_name else None
    self.model_ckpt = model_ckpt if model_ckpt else None
    self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.reload()

  def generate(self, *args, **kwargs):
    if not hasattr(self, "generator"):
        self.reload()
    return self.generator.generate(*args, **kwargs)

  def to(self, device):
    return self.generator.to(device)

  def reload(self):
    if hasattr(self, "generator"):
      del self.generator
    if self.device == "cuda":
      torch.cuda.empty_cache()
    self.generator = AttentionBasedGenerator(
        model_name=self.model_name,
        model_ckpt=self.model_ckpt,
        torch_dtype=self.dtype,
        cache_dir="cache",
    ).to(self.device)

generator = GeneratorWrapper(model_name, model_ckpt)


css = """
.btn-green {
  background-image: linear-gradient(to bottom right, #86efac, #22c55e) !important;
  border-color: #22c55e !important;
  color: #166534 !important;
}
.btn-green:hover {
  background-image: linear-gradient(to bottom right, #86efac, #86efac) !important;
}
.btn-red {
  background: linear-gradient(to bottom right, #fda4af, #fb7185) !important;
  border-color: #fb7185 !important;
  color: #9f1239 !important;
}
.btn-red:hover {background: linear-gradient(to bottom right, #fda4af, #fda4af) !important;}

/*****/

.dark .btn-green {
  background-image: linear-gradient(to bottom right, #047857, #065f46) !important;
  border-color: #047857 !important;
  color: #ffffff !important;
}
.dark .btn-green:hover {
  background-image: linear-gradient(to bottom right, #047857, #047857) !important;
}
.dark .btn-red {
  background: linear-gradient(to bottom right, #be123c, #9f1239) !important;
  border-color: #be123c !important;
  color: #ffffff !important;
}
.dark .btn-red:hover {background: linear-gradient(to bottom right, #be123c, #be123c) !important;}
"""

def generate_fn(
    feedback_enabled,
    max_feedback_imgs,
    image,
    prompt,
    neg_prompt,
    liked,
    disliked,
    denoising_steps,
    guidance_scale,
    feedback_start,
    feedback_end,
    min_weight,
    max_weight,
    neg_scale,
    batch_size,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
  try:
    if seed < 0:
      seed = None

    max_feedback_imgs = max(0, int(max_feedback_imgs))
    total_images = (len(liked) if liked else 0) + (len(disliked) if disliked else 0)

    if not feedback_enabled:
      liked = []
      disliked = []
    elif total_images > max_feedback_imgs:
      if liked and disliked:
        max_disliked = min(len(disliked), max_feedback_imgs // 2)
        max_liked = min(len(liked), max_feedback_imgs - max_disliked)
        if max_liked > len(liked):
          max_disliked = max_feedback_imgs - max_liked
        liked = liked[-max_liked:]
        disliked = disliked[-max_disliked:]
      elif liked:
        liked = liked[-max_feedback_imgs:]
        disliked = []
      else:
        liked = []
        disliked = disliked[-max_feedback_imgs:]
    # else: keep all feedback images
    
    generate_kwargs = {
        "prompt": prompt,
        "image": image,
        "negative_prompt": neg_prompt,
        "liked": liked,
        "disliked": disliked,
        "denoising_steps": denoising_steps,
        "guidance_scale": guidance_scale,
        "feedback_start": feedback_start,
        "feedback_end": feedback_end,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "neg_scale": neg_scale,
        "seed": seed,
        "n_images": batch_size,
    }

    try:
      images = generator.generate(**generate_kwargs)
    except RuntimeError as err:
      if 'out of memory' in str(err):
        generator.reload()
      raise
    return [(img, f"Image {i+1}") for i, img in enumerate(images)], images
  except Exception as err:
    raise gr.Error(str(err))


def add_img_from_list(i, curr_imgs, all_imgs):
  if all_imgs is None:
    all_imgs = []
  if i >= 0 and i < len(curr_imgs):
    all_imgs.append(curr_imgs[i])
  return all_imgs, all_imgs  # return (gallery, state)

def add_img(img, all_imgs):
  if all_imgs is None:
    all_imgs = []
  all_imgs.append(img)
  return None, all_imgs, all_imgs

def remove_img_from_list(event: gr.SelectData, imgs):
  if event.index >= 0 and event.index < len(imgs):
    imgs.pop(event.index)
  return imgs, imgs


with gr.Blocks(css=css) as demo:

  liked_imgs = gr.State([])
  disliked_imgs = gr.State([])
  curr_imgs = gr.State([])

  with gr.Row():
    with gr.Column(scale=100):
      prompt = gr.Textbox(label="Prompt")
      neg_prompt = gr.Textbox(label="Negative prompt", value="lowres, bad anatomy, bad hands, cropped, worst quality")
    # add upload image 
    image = gr.Image(type="pil", shape=(512, 512), label="Upload image")
    submit_btn = gr.Button("Generate", variant="primary", min_width="96px")

  with gr.Row(equal_height=False):
    with gr.Column():
      denoising_steps = gr.Slider(1, 100, value=20, step=1, label="Sampling steps")
      guidance_scale = gr.Slider(0.0, 30.0, value=6, step=0.25, label="CFG scale")
      batch_size = gr.Slider(1, 10, value=4, step=1, label="Batch size", interactive=False)
      seed = gr.Number(-1, minimum=-1, precision=0, label="Seed")
      max_feedback_imgs = gr.Slider(0, 20, value=6, step=1, label="Max. feedback images", info="Maximum number of liked/disliked images to be used. If exceeded, only the most recent images will be used as feedback. (NOTE: large number of feedback imgs => high VRAM requirements)")
      feedback_enabled = gr.Checkbox(True, label="Enable feedback", interactive=True)

      with gr.Accordion("Liked Images", open=True):
        liked_img_input = gr.Image(type="pil", shape=(512, 512), height=128, label="Upload liked image")
        like_gallery = gr.Gallery(label="👍 Liked images (click to remove)", columns=[3, 4, 3, 4, 5, 6], height=256, allow_preview=False)
        clear_liked_btn = gr.Button("Clear likes")

      with gr.Accordion("Disliked Images", open=True):
        disliked_img_input = gr.Image(type="pil", shape=(512, 512), height=128, label="Upload disliked image")
        dislike_gallery = gr.Gallery(label="👎 Disliked images (click to remove)", columns=[3, 4, 3, 4, 5, 6], height=256, allow_preview=False)
        clear_disliked_btn = gr.Button("Clear dislikes")

      with gr.Accordion("Feedback parameters", open=False):
        feedback_start = gr.Slider(0.0, 1.0, value=0.0, label="Feedback start", info="Fraction of denoising steps starting from which to use max. feedback weight.")
        feedback_end = gr.Slider(0.0, 1.0, value=0.8, label="Feedback end", info="Up to what fraction of denoising steps to use max. feedback weight.")
        feedback_min_weight = gr.Slider(0.0, 1.0, value=0.0, label="Feedback min. weight", info="Attention weight of feedback images when turned off (set to 0.0 to disable)")
        feedback_max_weight = gr.Slider(0.0, 1.0, value=0.8, label="Feedback max. weight", info="Attention weight of feedback images when turned on (set to 0.0 to disable)")
        feedback_neg_scale = gr.Slider(0.0, 1.0, value=0.5, label="Neg. feedback scale", info="Attention weight of disliked images relative to liked images (set to 0.0 to disable negative feedback)")

    with gr.Column():
      gallery = gr.Gallery(label="Generated images")

      like_btns = []
      dislike_btns = []
      with gr.Row():
        for i in range(0, 2):
          like_btn = gr.Button(f"👍 Image {i+1}", elem_classes="btn-green")
          like_btns.append(like_btn)
      with gr.Row():
        for i in range(2, 4):
          like_btn = gr.Button(f"👍 Image {i+1}", elem_classes="btn-green")
          like_btns.append(like_btn)
      with gr.Row():
        for i in range(0, 2):
          dislike_btn = gr.Button(f"👎 Image {i+1}", elem_classes="btn-red")
          dislike_btns.append(dislike_btn)
      with gr.Row():
        for i in range(2, 4):
          dislike_btn = gr.Button(f"👎 Image {i+1}", elem_classes="btn-red")
          dislike_btns.append(dislike_btn)

  generate_params = [
      feedback_enabled,
      max_feedback_imgs,
      image,
      prompt,
      neg_prompt,
      liked_imgs,
      disliked_imgs,
      denoising_steps,
      guidance_scale,
      feedback_start,
      feedback_end,
      feedback_min_weight,
      feedback_max_weight,
      feedback_neg_scale,
      batch_size,
      seed,
  ]
  submit_btn.click(generate_fn, generate_params, [gallery, curr_imgs], queue=True)

  for i, like_btn in enumerate(like_btns):
    like_btn.click(functools.partial(add_img_from_list, i), [curr_imgs, liked_imgs], [like_gallery, liked_imgs], queue=False)
  for i, dislike_btn in enumerate(dislike_btns):
    dislike_btn.click(functools.partial(add_img_from_list, i), [curr_imgs, disliked_imgs], [dislike_gallery, disliked_imgs], queue=False)

  like_gallery.select(remove_img_from_list, [liked_imgs], [like_gallery, liked_imgs], queue=False)
  dislike_gallery.select(remove_img_from_list, [disliked_imgs], [dislike_gallery, disliked_imgs], queue=False)

  liked_img_input.upload(add_img, [liked_img_input, liked_imgs], [liked_img_input, like_gallery, liked_imgs], queue=False)
  disliked_img_input.upload(add_img, [disliked_img_input, disliked_imgs], [disliked_img_input, dislike_gallery, disliked_imgs], queue=False)

  clear_liked_btn.click(lambda: [[], []], None, [liked_imgs, like_gallery], queue=False)
  clear_disliked_btn.click(lambda: [[], []], None, [disliked_imgs, dislike_gallery], queue=False)

demo.queue(1)
demo.launch(debug=True, share=True)