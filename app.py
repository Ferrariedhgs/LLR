import misc_f
import custom_styles

import gradio as gr
import spaces
import time
import numpy as np
from PIL import Image, ImageDraw
import json

# ─── Helpers ──────────────────────────────────────────────────────────────────

def placeholder(w, h, label="", r=15, g=55, b=55):
    img = Image.new("RGB", (w, h), (r, g, b))
    d = ImageDraw.Draw(img)
    # subtle grid
    for x in range(0, w, 32):
        d.line([(x, 0), (x, h)], fill=(0, 200, 150, 30), width=1)
    for y in range(0, h, 32):
        d.line([(0, y), (w, y)], fill=(0, 200, 150, 30), width=1)
    if label:
        d.text((w//2, h//2), label, fill=(0, 220, 170, 80), anchor="mm")
    return img


text_model=None
image_model=None
tts_model=None
room=""
key1=None
key2=None
door_key=None

@spaces.GPU
def run_generation():
    
    text_model, image_model, tts_model=misc_f.load_models()


    room=json.loads(misc_f.generate_game(text_model))


    rm  = misc_f.generate_image(model=image_model,prompt=room["room_prompt"],type="room")
    l0    = misc_f.generate_image(model=image_model,prompt=room["containers"][0]["container_prompt"],type="location")
    l1    = misc_f.generate_image(model=image_model,prompt=room["containers"][1]["container_prompt"],type="location")
    l2    = misc_f.generate_image(model=image_model,prompt=room["containers"][2]["container_prompt"],type="location")
    l3    = misc_f.generate_image(model=image_model,prompt=room["containers"][3]["container_prompt"],type="location")
    d   = misc_f.generate_image(model=image_model,prompt=room["door_prompt"],type="location")

    #key1=misc_f.generate_image(model=image_model,prompt=room["keys"][0]["key_prompt"],type="item")
    #key2=misc_f.generate_image(model=image_model,prompt=room["keys"][1]["key_prompt"],type="item")
    #door_key=misc_f.generate_image(model=image_model,prompt=room["door_key_prompt"],type="item")
    k0   = placeholder(200,  200,  "",   24, 88, 80)
    k1   = placeholder(200,  200,  "",   22, 82, 85)
    dk = placeholder(200,  200,  "", 19, 73, 67)

    return rm, l0, l1, l2, l3, d, k0, k1, dk



# ─── Build UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="1000 Rooms") as demo:

    # ── hidden state ──
    app_state = gr.State("generate")   # "generate" | "main"
    cont_count = gr.State(0)

    # ── loading overlay (always in DOM, shown/hidden by JS) ──
    gr.HTML("""
    <div id="nn-overlay" class="hidden">
        <div id="nn-label">Generating…</div>
        <canvas id="nn-canvas"></canvas>
    </div>
    """)

    # ══════════════════════════════════════════════════════
    #  GENERATE SCREEN
    # ══════════════════════════════════════════════════════
    with gr.Column(visible=True, elem_id="gen-screen") as gen_screen:
        gr.HTML("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:80vh;gap:2rem;">
            <div style="font-family:'Orbitron',sans-serif;font-size:2.2rem;
                        font-weight:900;color:#00ffcc;letter-spacing:0.25em;
                        text-shadow:0 0 30px rgba(0,255,204,0.4);">
                1000 Rooms
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
                        color:#4a9f8f;letter-spacing:0.15em;">
                Infinite escape rooms
            </div>
        </div>
        """)
        gen_btn = gr.Button("Generate", variant="primary", elem_id="gen-btn")

    # ══════════════════════════════════════════════════════
    #  MAIN SCREEN
    # ══════════════════════════════════════════════════════
    with gr.Column(visible=False, elem_id="main-screen") as main_screen:

        # ── TOP ROW ──
        with gr.Row():
            with gr.Column(scale=0, min_width=400):
                gr.HTML('<div class="hud">The Room</div>')
                main_img = gr.Image(
                    label="", show_label=False,
                    height=400, width=400,
                    interactive=False, elem_id="main-img"
                )

            with gr.Column(scale=0, min_width=400):
                gr.HTML('<div class="hud">Where is the Key?</div>')
                gallery_top = gr.Gallery(
                    label="", show_label=False,
                    columns=2, rows=2,
                    height=400,
                    object_fit="cover",
                    allow_preview=False,
                    fit_columns=False,
                    elem_id="top-gallery",
                    interactive=False,
                )

            with gr.Column(scale=0, min_width=400):
                gr.HTML('<div class="hud">Narrative</div>')
                narrative = gr.Textbox(
                    label="", show_label=False,
                    placeholder="The story...",
                    lines=17,
                    elem_id="narrative-box",
                    interactive=False,
                )

        # ── MIDDLE ROW ──
        with gr.Row():
            with gr.Column(scale=3, min_width=160):
                gr.HTML('<div class="hud">The Door</div>')
                mid_img = gr.Image(
                    label="", show_label=False,
                    height=200, width=200,
                    interactive=False, elem_id="mid-img"
                )

            with gr.Column(scale=3, min_width=400):
                gr.HTML('<div class="hud">Inventory</div>')
                stack_gallery = gr.Gallery(
                    label="", show_label=False,
                    columns=2, rows=1,
                    height=200,
                    object_fit="cover",
                    allow_preview=False,
                    fit_columns=False,
                    elem_id="stack-gallery",
                    interactive=False,
                )

            with gr.Column(scale=3, min_width=120):
                gr.HTML('<div class="hud">The Key</div>')
                small_img = gr.Image(
                    label="", show_label=False,
                    height=200, width=200,
                    interactive=False, elem_id="small-img"
                )

            with gr.Column(scale=4, min_width=160):
                gr.HTML('<div class="hud">Options</div>')
                listbox = gr.Radio(
                    choices=misc_f.VOICES, label="", show_label=False,
                    interactive=True,
                    value="Alfred",
                    elem_id="listbox",
                )

        # ── BOTTOM ROW ──
        with gr.Row():
            cont_btn = gr.Button("Continue →", variant="primary", scale=1, elem_id="cont-btn")

    # ══════════════════════════════════════════════════════
    #  EVENT WIRING
    # ══════════════════════════════════════════════════════

    # 1. Generate button – JS shows overlay first, Python does work, JS hides overlay
    def on_generate():
        main, g0, g1, g2, g3, mid, st0, st1, small, items = run_generation()
        return (
            gr.update(visible=False),                   # gen_screen
            gr.update(visible=True),                    # main_screen
            main,                                       # main_img
            [g0, g1, g2, g3],                           # top gallery
            mid,                                        # mid_img
            [st0, st1],                                 # stack gallery
            small,                                      # small_img
            #gr.update(choices=items, value=[]),         # listbox
            "",                                         # narrative
            0,                                          # cont_count reset
        )

    gen_btn.click(
        fn=on_generate,
        inputs=[],
        outputs=[gen_screen, main_screen, main_img, gallery_top,
                 mid_img, stack_gallery, small_img, listbox, narrative, cont_count],
        js="""() => {
            // Show overlay + start NN animation immediately (before server call)
            const ov = document.getElementById('nn-overlay');
            if (ov) ov.classList.remove('hidden');
            startNN();
            return [];
        }""",
    )


    # We attach the overlay-hide to main_img update instead (reliable trigger)
    main_img.change(
        fn=None,
        inputs=[],
        outputs=[],
        js="""() => {
            const ov = document.getElementById('nn-overlay');
            if (ov) ov.classList.add('hidden');
            stopNN();
        }"""
    )

    # 2. Continue button
    RESET_AFTER = 3   # change this to your actual condition

    def on_continue(text, count):
        count += 1
        new_text = (text or "") + f"\n[Turn {count}] …" if text else f"[Turn {count}] …"
        if count >= RESET_AFTER:
            # Reset to generate screen
            return (
                gr.update(visible=True),    # gen_screen
                gr.update(visible=False),   # main_screen
                "",                         # narrative cleared
                0,                          # count reset
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            new_text,
            count,
        )

    cont_btn.click(
        fn=on_continue,
        inputs=[narrative, cont_count],
        outputs=[gen_screen, main_screen, narrative, cont_count],
    )

    # Gallery selection – JS-side only (visual toggle, no server round-trip needed)
    gr.HTML("""
    <script>
    // Add click-to-select on gallery items after they render
    function attachGalSelect(galleryId) {
        const el = document.getElementById(galleryId);
        if (!el) return;
        el.querySelectorAll('.thumbnail-item, .gallery-item').forEach(item => {
            if (item.dataset.selBound) return;
            item.dataset.selBound = '1';
            item.addEventListener('click', function(e) {
                el.querySelectorAll('.selected-gallery').forEach(selected => {
                    selected.classList.remove('selected-gallery');
                });
                this.classList.add('selected-gallery');
                e.stopPropagation();
            });
        });
    }

    // Poll until gallery items appear (Gradio renders them lazily)
    function pollGalleries() {
        attachGalSelect('top-gallery');
        attachGalSelect('stack-gallery');
        setTimeout(pollGalleries, 1200);
    }
    document.addEventListener('DOMContentLoaded', pollGalleries);
    setTimeout(pollGalleries, 500);
    </script>
    """)

demo.launch(
    #server_name="0.0.0.0",
    #server_port=7860,
    css=custom_styles.CSS,
    head=custom_styles.HEAD_JS,
)
