---
name: demo-and-prototype-building
description: Building interactive demos and prototypes with Gradio, Streamlit, and similar frameworks for ML models and product concepts.
---

# Demo and Prototype Building

## When to Use This Skill

Use when building interactive demos for ML models, investor presentations, user testing prototypes, or internal tools. Covers framework selection, deployment, and common pitfalls.

## Demo Framework Selection

| Criteria | Gradio | Streamlit | Panel | Marimo |
|---|---|---|---|---|
| Best for | ML model demos | Data dashboards | Complex dashboards | Notebook-style apps |
| Learning curve | 15 minutes | 30 minutes | 1-2 hours | 30 minutes |
| State management | Stateless by default | Session state (tricky) | Param-based (robust) | Reactive cells |
| Sharing | Built-in share link | Streamlit Cloud | Manual deploy | Manual deploy |
| HF Spaces | First-class | Supported | No | No |
| Custom components | Yes (JS) | Yes (React) | Yes (Bokeh) | Limited |
| Real-time updates | Streaming support | st.empty() rerun | Periodic callbacks | Reactive |

### Decision Rule

- **ML model demo for external audience**: Gradio + HF Spaces. Shareable link in minutes.
- **Data dashboard / internal tool**: Streamlit. Better layout control, richer widgets.
- **Complex multi-page dashboard**: Panel. Proper state management, composable layouts.
- **Interactive notebook replacement**: Marimo. Reactive execution, reproducible.
- **Investor demo**: Gradio or Streamlit + custom CSS. Ship fast, look polished.

## Gradio Interface with File Upload

```python
import gradio as gr
from PIL import Image
import torch

model = load_model("my-model")

def classify_image(image: Image.Image, confidence_threshold: float) -> dict:
    tensor = preprocess(image)
    with torch.no_grad():
        outputs = model(tensor)
    probs = torch.softmax(outputs, dim=-1)
    return {LABELS[i]: float(p) for i, p in enumerate(probs[0]) if p.item() > confidence_threshold}

def summarize_text(text: str, max_length: int) -> str:
    if len(text.strip()) < 50:
        return "Input too short to summarize."
    return model.summarize(text, max_length=max_length)

image_demo = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(0.0, 1.0, value=0.5, label="Confidence Threshold"),
    ],
    outputs=gr.Label(num_top_classes=5),
    examples=[["examples/cat.jpg", 0.5], ["examples/dog.jpg", 0.3]],
    cache_examples=True,
)

text_demo = gr.Interface(
    fn=summarize_text,
    inputs=[
        gr.Textbox(lines=10, placeholder="Paste text here...", label="Input"),
        gr.Slider(50, 500, value=150, step=10, label="Max Length"),
    ],
    outputs=gr.Textbox(label="Summary"),
)

app = gr.TabbedInterface([image_demo, text_demo], ["Image", "Text"])
app.queue(max_size=20)
app.launch(server_name="0.0.0.0", server_port=7860)
```

## Streamlit Dashboard with Caching

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Startup Metrics", layout="wide")

@st.cache_data(ttl=300)
def load_metrics() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM daily_metrics", get_db_connection())

@st.cache_resource
def load_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased")

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(pd.Timestamp.now() - pd.Timedelta(days=30), pd.Timestamp.now()),
)
metric = st.sidebar.selectbox("Metric", ["revenue", "signups", "churn_rate"])

df = load_metrics()
mask = (df["date"] >= str(date_range[0])) & (df["date"] <= str(date_range[1]))
filtered = df[mask]

col1, col2, col3 = st.columns(3)
with col1:
    val = filtered[metric].iloc[-1] if len(filtered) > 0 else 0
    st.metric(metric.replace("_", " ").title(), f"${val:,.0f}")
with col2:
    st.metric("MRR", f"${filtered['mrr'].iloc[-1]:,.0f}" if len(filtered) > 0 else "$0")
with col3:
    st.metric("Active Users", f"{filtered['dau'].iloc[-1]:,}" if len(filtered) > 0 else "0")

fig = px.line(filtered, x="date", y=metric, title=f"{metric.title()} Over Time")
st.plotly_chart(fig, use_container_width=True)

st.header("Sentiment Analysis")
user_input = st.text_area("Enter text to analyze")
if st.button("Analyze") and user_input:
    result = load_model()(user_input)[0]
    st.success(f"**{result['label']}** (confidence: {result['score']:.2%})")
```

## Hugging Face Spaces Deployment

### README.md (Spaces Config)

```yaml
---
title: My ML Demo
emoji: "+"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
models:
  - my-org/my-model
---
```

### Deploy via Git

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create my-demo --type space --space-sdk gradio
git clone https://huggingface.co/spaces/my-username/my-demo
cp app.py requirements.txt my-demo/
cd my-demo && git add . && git commit -m "Initial demo" && git push
```

## Custom CSS Theming

### Gradio Custom Theme

```python
import gradio as gr

theme = gr.themes.Soft(
    primary_hue="blue", secondary_hue="slate", neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

custom_css = """
.gradio-container { max-width: 900px \!important; margin: auto; }
footer { display: none \!important; }
"""

with gr.Blocks(theme=theme, css=custom_css, title="Polished Demo") as demo:
    gr.Markdown("# Product Demo")
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(label="Input", lines=5)
            submit_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            output_text = gr.Textbox(label="Output", lines=8, interactive=False)
    submit_btn.click(fn=process, inputs=input_text, outputs=output_text)
```

### Streamlit Custom Theme

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#4F46E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1E293B"
font = "sans serif"
```

```python
# Hide Streamlit chrome for presentations
hide = "<style>#MainMenu,footer,header{visibility:hidden;}.block-container{padding-top:2rem;max-width:900px;}</style>"
st.markdown(hide, unsafe_allow_html=True)
```

## Gotchas and Anti-Patterns

### State Management in Streamlit
- **Problem**: Streamlit reruns the entire script on every interaction. Variables reset. Users lose form inputs.
- **Fix**: Use `st.session_state` for all persistent values. Initialize at script top. Use `st.form()` for multi-input submissions. Never put expensive ops outside `@st.cache_data`.
### Gradio Queue Configuration
- **Problem**: Multiple concurrent users without queuing causes failures/timeouts. Long inference blocks the app.
- **Fix**: Always call `app.queue(max_size=20, default_concurrency_limit=2)`. Set per-function `concurrency_limit` for GPU endpoints. Use `gr.Progress()` for long tasks.
### Demo vs Production Confusion
- **Problem**: Stakeholders see Gradio demo and assume it is production-ready. Gets shared externally without auth or rate limiting.
- **Fix**: Add visible "DEMO" banner. Use `share=False` for internal demos. Set `max_size` on queue. Add basic auth: `app.launch(auth=("user", "pass"))`.
### Large Model Loading in Serverless
- **Problem**: HF Spaces cold-starts take 60+ seconds loading a 3GB model. Users leave.
- **Fix**: Use quantized/distilled models for demos. Cache with `@st.cache_resource` or module-level load in Gradio. Show loading indicators. Consider GPU Spaces for large models.
### Forgetting to Pin Dependencies
- **Problem**: Demo breaks on deploy because latest Gradio introduced breaking changes overnight.
- **Fix**: Pin exact versions in `requirements.txt`. Test locally with pinned versions. Update intentionally.
