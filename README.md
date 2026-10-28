# 🎨 AI Text-to-Image Generator (Streamlit)

A modern, responsive Streamlit web application for generating images from text prompts using Hugging Face **Diffusers** and **Stable Diffusion**.

Converted and upgraded from the Jupyter Notebook `ipywidgets` implementation.

---

## 🚀 Features

- **Prompt Input & Suggestions**: Includes a prompt box (with the default prompt `"A beautiful mountain landscape at sunrise"`) and quick inspiration buttons.
- **Model Pipeline Caching**: Uses `@st.cache_resource` to load model weights once into memory.
- **Hardware Acceleration**: Automatic GPU (`CUDA`) and CPU detection with automatic dtype selection (`fp16` on GPU, `fp32` on CPU).
- **Model Options**:
  - `runwayml/stable-diffusion-v1-5` (Default standard Stable Diffusion)
  - `stabilityai/sd-turbo` (Fast 1-4 step generation, great for CPU/lighter hardware)
  - `CompVis/stable-diffusion-v1-4`
  - Custom Hugging Face Model ID
- **Hyperparameter Controls**:
  - Inference steps slider (User default: 25)
  - Guidance scale slider (User default: 7.5)
  - Dimensions (Width x Height: 256, 512, 768)
  - Seed control (Random or fixed integer)
  - Negative prompt (Optional exclusions)
- **Image Saving & Export**:
  - Automatically saves the latest generation locally as `generated_image.png` (matching your original code).
  - Browser download button to download the PNG image directly.
  - Session history gallery to review previous image generations.

---

## 📦 Installation

Ensure your Python environment is active and install the dependencies:

```bash
pip install -r requirements.txt
```

*(Or manually: `pip install streamlit diffusers transformers accelerate torch pillow`)*

---

## ▶️ Running the Application

In your terminal, run:

```bash
streamlit run app.py
```

Then open the local URL in your browser (typically `http://localhost:8501`).

---

## 🔄 Code Mapping: ipywidgets vs. Streamlit

| ipywidgets (Colab/Jupyter) | Streamlit (`app.py`) |
| :--- | :--- |
| `widgets.Text(value="A beautiful mountain...")` | `prompt = st.text_area(...)` |
| `widgets.Button(description="Generate Image")` | `st.button("🎨 Generate Image")` |
| `widgets.Output()`, `clear_output(wait=True)` | Native reactive Streamlit updates |
| `display(generated_image)` | `st.image(generated_image)` |
| `generated_image.save("generated_image.png")` | `generated_image.save("generated_image.png")` + direct `st.download_button` |
| Manual script execution | Cached pipeline with `@st.cache_resource` & hardware detection |
