import os
import io
import time
import streamlit as st
from PIL import Image
import torch

# Configure Streamlit page
st.set_page_config(
    page_title="AI Text-to-Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Main container and typography */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header hero block */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 0;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-gpu {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-cpu {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    
    /* Card design */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    /* Image container */
    .image-preview-box {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Determine hardware device and dtype
def get_device_info():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        dtype = torch.float32
        device_name = "CPU (Inference may take 1-3 minutes)"
    return device, dtype, device_name


DEVICE, DTYPE, DEVICE_NAME = get_device_info()


# Cached model pipeline loader
@st.cache_resource(show_spinner=False)
def load_diffusion_pipeline(model_id: str):
    """
    Loads and caches the diffusion model pipeline in memory.
    """
    try:
        from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
        
        # Load pipeline based on model type
        if "turbo" in model_id.lower():
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=DTYPE,
                variant="fp16" if DEVICE == "cuda" else None
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=DTYPE,
                safety_checker=None  # Optional: keeps memory usage lower
            )
            
        pipe = pipe.to(DEVICE)
        
        # CPU memory optimizations
        if DEVICE == "cpu":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
                
        return pipe, None
    except Exception as e:
        return None, str(e)


# Initialize Session State
if "last_image" not in st.session_state:
    st.session_state.last_image = None
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "generation_time" not in st.session_state:
    st.session_state.generation_time = None
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- SIDEBAR: PARAMETERS & SETTINGS ----------------
with st.sidebar:
    st.title("⚙️ Model Settings")
    
    # Device status indicator
    if DEVICE == "cuda":
        st.markdown(f'<span class="badge badge-gpu">🟢 GPU Mode</span> <small>{DEVICE_NAME}</small>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge badge-cpu">🟡 CPU Mode</span> <br><small style="color:#94a3b8;">{DEVICE_NAME}</small>', unsafe_allow_html=True)
        st.caption("💡 *Tip: For fast generation on CPU or light GPUs, choose SD-Turbo.*")

    st.markdown("---")

    # Model Selection
    model_choice = st.selectbox(
        "Choose Model:",
        options=[
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/sd-turbo",
            "CompVis/stable-diffusion-v1-4",
            "Custom Hugging Face Model ID"
        ],
        index=0,
        help="Select a text-to-image model. SD 1.5 is the standard model; SD-Turbo is optimized for ultra-fast generation."
    )

    if model_choice == "Custom Hugging Face Model ID":
        selected_model = st.text_input("Enter HuggingFace Model Repo:", value="runwayml/stable-diffusion-v1-5")
    else:
        selected_model = model_choice

    st.markdown("### 🎛️ Hyperparameters")
    
    # Inference steps (user's default: 25)
    default_steps = 4 if "turbo" in selected_model.lower() else 25
    steps = st.slider(
        "Number of Inference Steps:",
        min_value=1,
        max_value=50,
        value=default_steps,
        step=1,
        help="Higher steps increase detail but take longer. User default is 25."
    )

    # Guidance scale (user's default: 7.5)
    default_guidance = 1.5 if "turbo" in selected_model.lower() else 7.5
    guidance = st.slider(
        "Guidance Scale (CFG):",
        min_value=1.0,
        max_value=15.0,
        value=float(default_guidance),
        step=0.5,
        help="Controls how strictly the model adheres to your text prompt. User default is 7.5."
    )

    # Image Dimensions
    col_w, col_h = st.columns(2)
    with col_w:
        width = st.selectbox("Width", options=[256, 512, 768], index=1)
    with col_h:
        height = st.selectbox("Height", options=[256, 512, 768], index=1)

    # Seed for reproducibility
    use_random_seed = st.checkbox("Randomize Seed", value=True)
    if use_random_seed:
        seed = int(torch.randint(0, 1000000, (1,)).item())
    else:
        seed = st.number_input("Custom Seed:", value=42, step=1)

    st.markdown("---")
    # Quick clear button
    if st.button("🧹 Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_image = None
        st.rerun()


# ---------------- MAIN CONTENT AREA ----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">✨ Text to Image Studio</div>
    <p class="hero-subtitle">Transform your imagination into visuals using Stable Diffusion and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Prompt Suggestions
st.markdown("**💡 Quick Prompt Ideas:**")
sample_prompts = [
    "A beautiful mountain landscape at sunrise",
    "Futuristic cyberpunk city at night with neon reflections, 8k",
    "A cute fluffy red panda drinking boba tea, digital art style",
    "Oil painting of an astronaut on a horseback in vibrant cosmos"
]

selected_sample = None
sample_cols = st.columns(len(sample_prompts))
for idx, prompt_text in enumerate(sample_prompts):
    if sample_cols[idx].button(prompt_text[:28] + "...", key=f"sample_{idx}", help=prompt_text):
        st.session_state["prompt_input"] = prompt_text

# Text input for prompt
prompt_value = st.session_state.get("prompt_input", "A beautiful mountain landscape at sunrise")
prompt = st.text_area(
    "Enter your prompt:",
    value=prompt_value,
    placeholder="Describe the image you want to generate in detail...",
    height=90,
    key="current_prompt"
)

# Optional Negative Prompt
with st.expander("➕ Negative Prompt (Optional)"):
    negative_prompt = st.text_area(
        "What to exclude from image:",
        value="blurry, distorted, low quality, artifacts, watermark",
        placeholder="e.g. blurry, cartoon, low resolution..."
    )

# Action button
col_gen, col_info = st.columns([1, 2])
with col_gen:
    generate_clicked = st.button("🎨 Generate Image", type="primary", use_container_width=True)

# ---------------- GENERATION LOGIC ----------------
if generate_clicked:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        st.warning("⚠️ Please enter a prompt before generating.")
    else:
        # Load model pipeline
        with st.status("🔄 Initializing Stable Diffusion pipeline...", expanded=True) as status:
            st.write(f"Loading weights for `{selected_model}` on **{DEVICE.upper()}**...")
            pipe, err = load_diffusion_pipeline(selected_model)
            
            if err:
                status.update(label="❌ Model Loading Failed", state="error", expanded=True)
                st.error(f"Error loading model pipeline: {err}")
                st.info("Tip: Ensure your machine has an active internet connection to download Hugging Face weights on first run, or try a smaller model like `stabilityai/sd-turbo`.")
            else:
                status.update(label="⚡ Pipeline ready! Generating image...", state="running")
                st.write(f"Running inference with {steps} steps and guidance scale {guidance}...")
                
                # Setup generator for deterministic seed
                generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
                
                start_time = time.time()
                try:
                    # Execute diffusion inference
                    # Turbo models do not require guidance_scale or negative_prompt
                    if "turbo" in selected_model.lower():
                        result = pipe(
                            prompt=clean_prompt,
                            num_inference_steps=steps,
                            guidance_scale=0.0,
                            width=width,
                            height=height,
                            generator=generator
                        )
                    else:
                        result = pipe(
                            prompt=clean_prompt,
                            negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            width=width,
                            height=height,
                            generator=generator
                        )
                    
                    elapsed = time.time() - start_time
                    generated_image = result.images[0]
                    
                    # 1. Save locally as generated_image.png (as in user's original code)
                    output_filename = "generated_image.png"
                    generated_image.save(output_filename)
                    
                    # 2. Update session state
                    st.session_state.last_image = generated_image
                    st.session_state.last_prompt = clean_prompt
                    st.session_state.generation_time = elapsed
                    st.session_state.history.insert(0, {
                        "prompt": clean_prompt,
                        "image": generated_image,
                        "time": elapsed,
                        "seed": seed
                    })
                    
                    status.update(label=f"✅ Image generated successfully in {elapsed:.1f}s!", state="complete")
                    st.toast("🎉 Image generated & saved as generated_image.png!", icon="🖼️")
                    
                except Exception as ex:
                    status.update(label="❌ Generation error", state="error")
                    st.error(f"An error occurred during generation: {str(ex)}")


# ---------------- DISPLAY RESULT ----------------
if st.session_state.last_image is not None:
    st.markdown("---")
    st.markdown("### 🖼️ Generated Output")
    
    col_img, col_details = st.columns([3, 2], gap="medium")
    
    with col_img:
        # Display the generated image
        st.image(
            st.session_state.last_image,
            caption=f"Prompt: {st.session_state.last_prompt}",
            use_container_width=True
        )
        
    with col_details:
        st.success(f"💾 **Saved to disk:** `generated_image.png`")
        if st.session_state.generation_time:
            st.info(f"⏱️ **Inference Time:** {st.session_state.generation_time:.2f} seconds")
            
        # Download button for convenient direct saving from browser
        img_buffer = io.BytesIO()
        st.session_state.last_image.save(img_buffer, format="PNG")
        st.download_button(
            label="📥 Download PNG Image",
            data=img_buffer.getvalue(),
            file_name=f"generated_{int(time.time())}.png",
            mime="image/png",
            use_container_width=True
        )
        
        with st.expander("📋 Generation Metadata", expanded=True):
            st.write(f"**Model:** `{selected_model}`")
            st.write(f"**Inference Steps:** `{steps}`")
            st.write(f"**Guidance Scale:** `{guidance}`")
            st.write(f"**Dimensions:** `{width} x {height}`")
            st.write(f"**Device:** `{DEVICE.upper()}`")
            st.write(f"**Seed:** `{seed}`")

# ---------------- HISTORY GALLERY ----------------
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown("### 🕒 Recent Generations in this Session")
    hist_cols = st.columns(min(4, len(st.session_state.history) - 1))
    for i, item in enumerate(st.session_state.history[1:5]):
        with hist_cols[i % len(hist_cols)]:
            st.image(item["image"], use_container_width=True)
            st.caption(f"_{item['prompt'][:35]}..._ ({item['time']:.1f}s)")
