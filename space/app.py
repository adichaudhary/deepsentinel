"""DeepSentinel Gradio demo for Hugging Face Spaces."""

import gradio as gr

from inference import DeepSentinel

detector = DeepSentinel()


def analyze(image):
    if image is None:
        return None, None, "Upload an image to begin."
    result = detector.predict(image, with_heatmap=True)
    scores = {"Fake": result.p_fake, "Real": round(1 - result.p_fake, 4)}
    note = "" if result.face_detected else "No face detected. The center of the image was analyzed instead."
    return scores, result.heatmap, note


with gr.Blocks(title="DeepSentinel") as demo:
    gr.Markdown(
        "# DeepSentinel\n"
        "Deepfake face detection with a fine-tuned EfficientNet-B4. "
        "Upload a photo with a face to get a prediction and a Grad-CAM map of the regions that drove it.\n\n"
        "[GitHub](https://github.com/adichaudhary/deepsentinel)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            scores = gr.Label(num_top_classes=2, label="Prediction")
            cam = gr.Image(type="pil", label="Grad-CAM")
            note = gr.Markdown()
    btn.click(analyze, inputs=inp, outputs=[scores, cam, note])
    inp.upload(analyze, inputs=inp, outputs=[scores, cam, note])

if __name__ == "__main__":
    demo.launch()
