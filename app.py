import torch
import einops
import gradio as gr
import datetime
import numpy as np
import spaces
import soundfile
import os
import sys
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.append("sf-creator-fork")
from main import sfz, decentsampler

decoder_path = "erl-j/soundfont-generator-assets/decoder.pt"
model_path = "erl-j/soundfont-generator-assets/synth_lfm_modern_bfloat16.pt"
# Download models from Hugging Face Hub
decoder_path = hf_hub_download("erl-j/soundfont-generator-assets", "decoder.pt")
model_path = hf_hub_download(
    "erl-j/soundfont-generator-assets", "synth_lfm_modern_bfloat16.pt"
)

# Load models once at startup
device = "cuda"
decoder = torch.load(decoder_path, map_location=device).half().eval()
model = torch.load(model_path, map_location=device).half().eval()


@spaces.GPU
def generate_and_export_soundfont(text, steps=20, instrument_name=None):
    sample_start = datetime.datetime.now()

    # Generate audio as before
    z = model.sample(1, text=[text], steps=steps)
    z_reshaped = einops.rearrange(z, "b t c d -> (b c) d t")

    with torch.no_grad():
        audio = decoder.decode(z_reshaped)

    audio_output = einops.rearrange(audio, "b c t -> c (b t)").cpu().numpy()
    audio_output = audio_output / np.max(np.abs(audio_output))

    # Export individual wav files
    export_audio = audio.cpu().numpy().astype(np.float32)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Create instrument name if not provided
    if not instrument_name:
        instrument_name = text.replace(" ", "_")[:20]

    # Save individual WAV files
    pitches = [
        "C1",
        "F#1",
        "C2",
        "F#2",
        "C3",
        "F#3",
        "C4",
        "F#4",
        "C5",
        "F#5",
        "C6",
        "F#6",
        "C7",
        "F#7",
        "C8",
    ]
    wav_files = []
    for i in range(audio.shape[0]):
        wav_path = f"{output_dir}/{pitches[i]}.wav"
        soundfile.write(wav_path, export_audio[i].T, 44100)
        wav_files.append(wav_path)

    # Generate SFZ file
    sfz(
        directory=output_dir,
        lowkey="21",
        highkey="108",
        instrument=instrument_name,
        loopmode="no_loop",
        polyphony=None,
    )

    # Create zip file containing SFZ and WAV files for the complete soundfont
    zip_path = f"{output_dir}/{instrument_name}_package.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        # Add SFZ file
        sfz_file = f"{output_dir}/{instrument_name}.sfz"
        zipf.write(sfz_file, os.path.basename(sfz_file))
        # Add all WAV files
        for wav_file in wav_files:
            if os.path.exists(wav_file):
                zipf.write(wav_file, os.path.basename(wav_file))

    total_time = (datetime.datetime.now() - sample_start).total_seconds()

    return (
        (44100, audio_output.T),
        f"Generation took {total_time:.2f}s\nFiles saved in {output_dir}",
        zip_path,
        wav_files,
    )


custom_js = open("custom.js").read()
custom_css = open("custom.css").read()

demo = gr.Blocks(
    title="Erl-j's Soundfont Generator",
    theme=gr.themes.Default(
        primary_hue="green",
        font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"],
    ),
    js=custom_js,
    css=custom_css,
)

with demo:
    gr.Markdown(open("intro.md").read())

    with gr.Row():
        steps = gr.Slider(
            minimum=1, maximum=50, value=20, step=1, label="Generation steps"
        )

    with gr.Row():
        text_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter text description (e.g. 'hard bass', 'sparkly bells')",
            lines=2,
        )

    with gr.Row():
        generate_btn = gr.Button("Generate Soundfont", variant="primary")

    with gr.Row():
        audio_output = gr.Audio(label="Generated Audio Preview", visible=False)
        status_output = gr.Textbox(label="Status", lines=2, visible=False)

    with gr.Row():
        wav_files = gr.File(
            label="Individual WAV Files",
            file_count="multiple",
            visible=False,
            elem_id="individual-wav-files",
        )

    html = """
    <div id="custom-player"
    style="width: 100%; height: 600px; border: 1px solid #f8f9fa; border-radius: 5px; margin-top: 10px;"
    ></div>
    """

    gr.HTML(html, min_height=1000, max_height=1000)

    gr.Markdown("## Download Soundfont Package here:")
    with gr.Row():
        sf = gr.File(
            label="Download SFZ Soundfont Package",
            type="filepath",
            visible=True,
            elem_id="sfz",
        )

    gr.Markdown("""
    # About            
    The model is a modified version of [stable audio open](https://huggingface.co/stabilityai/stable-audio-open-1.0).
                
    Unlike the original model, this version uses latent flow matching rather than latent diffusion.
    Secondly, the pitches are stacked in a channel dimension rather than concatenated in the time dimension. 
    This allows for faster generation.
                
    Soundfont export code is based on the [sf-creator](https://github.com/paulwellnerbou/sf-creator) project.
                
    Similar work by Nercessian and Imort: [InstrumentGen](https://instrumentgen.netlify.app/).
                
    Thank you @carlthome for coming up with the name.
                
    To cite this work, please use the following BibTeX entry:
    ```bibtex   
    @misc{erl-j-soundfont-generator,
        title={Erl-j's Soundfont Generator},
        author={Nicolas Jonason},
        year={2024},
        publisher={Huggingface},
    }
    ```
    """)

    generate_btn.click(
        fn=generate_and_export_soundfont,
        inputs=[text_input, steps],
        outputs=[audio_output, status_output, sf, wav_files],
    ).success(js="() => console.log('Success')")

    text_input.submit(
        fn=generate_and_export_soundfont,
        inputs=[text_input, steps],
        outputs=[audio_output, status_output, sf, wav_files],
    )

if __name__ == "__main__":
    print("Starting demo...")
    demo.launch()
