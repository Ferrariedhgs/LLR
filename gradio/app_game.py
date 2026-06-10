# gradio and spaces must be imported before misc_f_game on ZeroGPU
import spaces
import gradio as gr

# misc_f_game sets torch._dynamo.config.disable=True at import time
import misc_f_game

import logging
import numpy as np
from PIL import Image, ImageDraw
import random

import custom_styles

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─── ZeroGPU model strategy ───────────────────────────────────────────────────
# ZeroGPU evicts GPU memory between @spaces.GPU frames — models loaded in one
# frame are NOT available in the next. The only reliable pattern is:
#   load the model you need at the TOP of each GPU function, use it, done.
#
# To avoid re-downloading from HuggingFace on every call we use the HF cache
# (models are on disk after the first download) — loading from local cache is
# fast (~3-8s for GGUF, ~10s for FLUX from disk).
#
# _text_loaded / _image_loaded / _tts_loaded are CPU-side flags so we can log
# "first load vs cache hit" — they do NOT hold model objects across frames.

_text_loaded  = False   # True after first successful text model download
_image_loaded = False   # True after first successful image model download
_tts_loaded   = False   # True after first successful TTS model download


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _placeholder(w, h, r=15, g=55, b=55):
    img = Image.new("RGB", (w, h), (r, g, b))
    d   = ImageDraw.Draw(img)
    for x in range(0, w, 32):
        d.line([(x, 0), (x, h)], fill=(0, 200, 150, 30), width=1)
    for y in range(0, h, 32):
        d.line([(0, y), (w, y)], fill=(0, 200, 150, 30), width=1)
    return img


narrator_selected="Alfred"
container_selected=0
key_selected=0
container_key_order=[]

def build_escape_chain(boxes, keys):
    random.shuffle(boxes)
    random.shuffle(keys)

    chain = []
    current_key = None

    for i, box in enumerate(boxes):
        if i < len(keys):
            next_key = keys[i]
        else:
            next_key = None

        chain.append({
            "container": box,
            "key": current_key,
            "contains": next_key
        })

        current_key = next_key

    return chain


# ─── Warmup — pre-downloads model weights to the HF disk cache ───────────────
# This does NOT keep models in GPU memory (ZeroGPU would evict them anyway).
# Its only job is to ensure weights are on disk so every subsequent load is a
# fast local read rather than a network download.
# Returns a status string so demo.load can update a gr.Textbox / gr.State.

@spaces.GPU(duration=300)
def warmup_models():
    global _text_loaded, _image_loaded
    log.info("Warmup: downloading/verifying model weights...")
    # Load then immediately discard — just to populate the HF cache on disk.
    misc_f_game.load_text_model()
    _text_loaded = True
    log.info("Warmup: text weights cached.")
    misc_f_game.load_image_model()
    _image_loaded = True
    log.info("Warmup: image weights cached. Ready.")
    return "ready"




@spaces.GPU(duration=600)
def run_generation():
    """
    Full generation in one GPU frame: room JSON then all 9 images.
    Returns (room, room_img, loc0-3, door_img, key0, key1, door_key_img).
    Keeping text + image in one frame avoids ZeroGPU token expiry between
    the two @spaces.GPU calls that the previous two-phase design used.
    """
    # ── Text ──────────────────────────────────────────────────────────────────
    log.info("run_generation: loading text model...")
    #text_model = misc_f_game.load_text_model()
    room = misc_f_game.generate_game(0)
    log.info("Room generated: %s", room.get("room_name"))
    #del text_model   # free VRAM before loading FLUX

    # ── Images ────────────────────────────────────────────────────────────────
    log.info("run_generation: loading image model...")
    #image_model = misc_f_game.load_image_model()

    room_img     = misc_f_game.generate_image(0, room["room_prompt"],                               "room")
    loc0         = misc_f_game.generate_image(0, room["containers"][0]["container_prompt"],         "location")
    loc1         = misc_f_game.generate_image(0, room["containers"][1]["container_prompt"],         "location")
    loc2         = misc_f_game.generate_image(0, room["containers"][2]["container_prompt"],         "location")
    loc3         = misc_f_game.generate_image(0, room["containers"][3]["container_prompt"],         "location")
    door_img     = misc_f_game.generate_image(0, room["door_prompt"],                               "location")
    key0         = misc_f_game.generate_image(0, room["keys"][0]["key_prompt"],                     "item")
    key1         = misc_f_game.generate_image(0, room["keys"][1]["key_prompt"],                     "item")
    door_key_img = misc_f_game.generate_image(0, room["door_key_prompt"],                           "item")

    containers=[cont["container_name"] for cont in room["containers"]]
    keys=[k["key_name"] for k in room["keys"]]
    container_key_order=build_escape_chain(containers,keys)

    return room, room_img, loc0, loc1, loc2, loc3, door_img, key0, key1, door_key_img, container_key_order


@spaces.GPU(duration=60)
def run_continue(container: str, key: str, right_key: bool, item_given: str = ""):
    """Narrate a container-opening attempt."""
    #text_model = misc_f_game.load_text_model()
    result = misc_f_game.continue_game(0, container, key, right_key, item_given)
    return result.get("text", "")


@spaces.GPU(duration=60)
def run_open_door(key: str, key_type: str):
    """Narrate a door-opening attempt."""
    #text_model = misc_f_game.load_text_model()
    result = misc_f_game.open_door(0, key, key_type)
    return result.get("text", "")


@spaces.GPU(duration=60)
def run_narration(narrator: str, prompt: str):
    """Synthesise voice narration."""
    log.info("run_narration: loading TTS model...")
    tts_model = misc_f_game.load_tts_model()
    return misc_f_game.generate_voice(tts_model, prompt, narrator)


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="1000 Rooms") as demo:

    room_state = gr.State(None)
    cont_count = gr.State(0)
    puzzle_order=gr.State(None)

    gr.HTML("""
    <div id="nn-overlay" class="hidden">
        <div id="nn-label">Generating…</div>
        <canvas id="nn-canvas"></canvas>
    </div>
    """)

    # ── Generate screen ──
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
        # Starts disabled; demo.load re-enables it when warmup_models returns.
        gen_btn    = gr.Button("Loading models…", variant="primary",
                               elem_id="gen-btn", interactive=False)
        status_lbl = gr.HTML(
            '<p style="text-align:center;color:#4a9f8f;font-size:0.75rem;'
            'font-family:\'Share Tech Mono\',monospace;margin-top:-0.5rem;">'
            'Downloading model weights, please wait…</p>',
            elem_id="status-lbl",
        )

    # ── Main screen ──
    with gr.Column(visible=False, elem_id="main-screen") as main_screen:

        with gr.Row():
            with gr.Column(scale=0, min_width=400):
                gr.HTML('<div class="hud">The Room</div>')
                main_img = gr.Image(
                    label="", show_label=False,
                    height=400, width=400,
                    interactive=False, elem_id="main-img",
                )

            with gr.Column(scale=0, min_width=400):
                gr.HTML('<div class="hud">Where is the Key?</div>')
                gallery_top = gr.Gallery(
                    label="", show_label=False,
                    columns=2, rows=2, height=400,
                    object_fit="cover", allow_preview=False,
                    fit_columns=False, elem_id="top-gallery", interactive=False,
                )

            with gr.Column(scale=0, min_width=400):
                gr.HTML('<div class="hud">Narrative</div>')
                narrative = gr.Textbox(
                    label="", show_label=False,
                    placeholder="The story...",
                    lines=17, elem_id="narrative-box", interactive=False,
                )

        with gr.Row():
            with gr.Column(scale=3, min_width=160):
                gr.HTML('<div class="hud">The Door</div>')
                mid_img = gr.Image(
                    label="", show_label=False,
                    height=200, width=200,
                    interactive=False, elem_id="mid-img",
                )

            with gr.Column(scale=3, min_width=600):
                gr.HTML('<div class="hud">Inventory</div>')
                stack_gallery = gr.Gallery(
                    label="", show_label=False,
                    columns=3, rows=1, height=200,
                    object_fit="cover", allow_preview=False,
                    fit_columns=False, elem_id="stack-gallery", interactive=False,
                )
            '''
            with gr.Column(scale=3, min_width=120):
                gr.HTML('<div class="hud">The Key</div>')
                small_img = gr.Image(
                    label="", show_label=False,
                    height=200, width=200,
                    interactive=False, elem_id="small-img",
                )
            '''
            with gr.Column(scale=4, min_width=160):
                gr.HTML('<div class="hud">Narrator</div>')
                listbox = gr.Radio(
                    choices=list(misc_f_game.VOICES.keys()),
                    label="", show_label=False,
                    interactive=True, value="Alfred", elem_id="listbox",
                )

        with gr.Row():
            cont_btn = gr.Button("Continue →", variant="primary", scale=1, elem_id="cont-btn")

    # ── Event wiring ──

    def on_warmup_complete(_status):
        """
        Called when demo.load finishes. Enables Generate and hides the
        status label regardless of whether warmup succeeded or failed.
        warmup_models() returns "ready" on success; on exception Gradio
        passes None, so we enable the button either way so the user isn't
        permanently locked out.
        """
        return (
            gr.update(value="Generate", interactive=True),   # gen_btn
            gr.update(visible=False),                        # status_lbl
        )

    


    def on_generate():
        """
        Single-phase generation: one GPU frame handles text + all 9 images.
        No yield needed — everything arrives together, no partial state.
        """
        (room, room_img, loc0, loc1, loc2, loc3,
         door_img, key0, key1, door_key_img, ck_o) = run_generation()

        story = room.get("room_story", "")
        return (
            gr.update(visible=False),       # gen_screen
            gr.update(visible=True),        # main_screen
            room_img,                       # main_img
            [loc0, loc1, loc2, loc3],       # gallery_top
            door_img,                       # mid_img
            [key0, key1, door_key_img],                   # stack_gallery                  # small_img
            story,                          # narrative
            0,                             # cont_count
            room,                          # room_state
            ck_o                           # puzzle order
        )

    # warmup_models returns a string; wire it through a hidden State so
    # on_warmup_complete can enable the button when it finishes.
    _warmup_result = gr.State(None)

    demo.load(
        fn=warmup_models,
        inputs=[],
        outputs=[_warmup_result],
    ).then(
        fn=on_warmup_complete,
        inputs=[_warmup_result],
        outputs=[gen_btn, status_lbl],
    )

    gen_btn.click(
        fn=on_generate,
        inputs=[],
        outputs=[
            gen_screen, main_screen,
            main_img, gallery_top, mid_img,
            stack_gallery,
            narrative, cont_count, room_state, puzzle_order
        ],
        js="""() => {
            const ov = document.getElementById('nn-overlay');
            if (ov) ov.classList.remove('hidden');
            startNN();
            return [];
        }""",
    )

    main_img.change(
        fn=None, inputs=[], outputs=[],
        js="""() => {
            const ov = document.getElementById('nn-overlay');
            if (ov) ov.classList.add('hidden');
            stopNN();
        }""",
    )

    def on_radio_change(nr):
        global narrator_selected
        narrator_selected=nr
    

    listbox.change(
        fn=on_radio_change,
        inputs=listbox,
    )

    def cont_gal_change(gal, evt: gr.SelectData):
        global container_selected
        container_selected = evt.index
        #container_selected = gal[index]
        
    gallery_top.select(
        fn=cont_gal_change,
        inputs=gallery_top,
    )

    def key_gal_change(gal, evt: gr.SelectData):
        global key_selected
        key_selected = evt.index
        #key_selected = gal[index]
        
    stack_gallery.select(
        fn=key_gal_change,
        inputs=stack_gallery,
    )


    RESET_AFTER = 50

    def on_continue(text, count, room):
        count   += 1
        new_text = f"{text}\n[Container: {container_selected}; key: {key_selected}; narrator: {narrator_selected}]\n" if text else f"{text}\n[Container: {container_selected}; key: {key_selected}; narrator: {narrator_selected}]\n"
        if count >= RESET_AFTER:
            return (
                gr.update(visible=True), gr.update(visible=False),
                "", 0, None,
            )
        return (
            gr.update(visible=False), gr.update(visible=True),
            new_text, count, room,
        )

    cont_btn.click(
        fn=on_continue,
        inputs=[narrative, cont_count, room_state],
        outputs=[gen_screen, main_screen, narrative, cont_count, room_state],
    )

    gr.HTML("""
    <script>
    function attachGalSelect(galleryId) {
        const el = document.getElementById(galleryId);
        if (!el) return;
        el.querySelectorAll('.thumbnail-item, .gallery-item').forEach(item => {
            if (item.dataset.selBound) return;
            item.dataset.selBound = '1';
            item.addEventListener('click', function(e) {
                el.querySelectorAll('.selected-gallery')
                  .forEach(s => s.classList.remove('selected-gallery'));
                this.classList.add('selected-gallery');
                e.stopPropagation();
            });
        });
    }
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
    css=custom_styles.CSS,
    head=custom_styles.HEAD_JS,
)
