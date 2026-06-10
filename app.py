# gradio and spaces must be imported before misc_f on ZeroGPU
import spaces
import gradio as gr

# misc_f sets torch._dynamo.config.disable=True at import time
import misc_f

import logging
import random
from PIL import Image, ImageDraw

import custom_styles

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─── ZeroGPU model strategy ───────────────────────────────────────────────────
# Text model  → CPU singleton in misc_f, never needs a GPU frame.
# Image model → @spaces.GPU, loaded fresh each frame from HF disk cache.
# TTS model   → @spaces.GPU, loaded fresh each frame from HF disk cache.
# _*_loaded flags are informational only (log "first load vs cache hit").

_image_loaded = False
_tts_loaded   = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _placeholder(w, h, r=15, g=55, b=55):
    img = Image.new("RGB", (w, h), (r, g, b))
    d   = ImageDraw.Draw(img)
    for x in range(0, w, 32):
        d.line([(x, 0), (x, h)], fill=(0, 200, 150, 30), width=1)
    for y in range(0, h, 32):
        d.line([(0, y), (w, y)], fill=(0, 200, 150, 30), width=1)
    return img


# ─── Chain logic ──────────────────────────────────────────────────────────────
# 4 containers, 3 keys (key0, key1, door_key).
#   chain[0]: open container  (opened_by=None),     contains keys[0]
#   chain[1]: locked,          opened_by=keys[0],   contains keys[1]
#   chain[2]: locked,          opened_by=keys[1],   contains door_key
#   chain[3]: locked,          opened_by=door_key,  contains None  ← red herring

def generate_random_chain(boxes, keys):
    boxes = boxes[:]
    keys  = keys[:]
    random.shuffle(boxes)
    random.shuffle(keys)
    chain = []
    for i, box in enumerate(boxes):
        chain.append({
            "box":       box,
            "opened_by": None    if i == 0 else keys[i - 1],
            "contains":  keys[i] if i < len(keys) else None,
        })
    return chain


# ─── Warmup ───────────────────────────────────────────────────────────────────
# Pre-downloads FLUX and TTS weights to the HF disk cache inside a GPU frame
# so subsequent loads are fast local reads.  Text model is CPU-only; we just
# ensure the GGUF file is on disk here — actual Llama() construction is deferred
# to the first inference call (outside any GPU frame).

@spaces.GPU(duration=120)
def warmup_models():
    global _image_loaded, _tts_loaded
    log.info("Warmup: pre-caching model weights...")
    misc_f._ensure_gguf()
    log.info("Warmup: GGUF on disk.")
    misc_f.load_image_model()
    _image_loaded = True
    log.info("Warmup: FLUX cached.")
    misc_f.load_tts_model()
    _tts_loaded = True
    log.info("Warmup: TTS cached. Ready.")
    return "ready"


# ─── Generation — two phases ──────────────────────────────────────────────────
# Phase 1 — CPU, no @spaces.GPU.
#   Generates room JSON using the CPU text singleton.
#   Runs while the nn-overlay animation is shown.
#
# Phase 2 — @spaces.GPU.
#   Generates all 9 images with FLUX.
#   Only this step consumes ZeroGPU budget.
#   Returns everything needed to populate the UI.

def run_text_generation(narrator: str) -> dict:
    """
    Phase 1: CPU-only room JSON generation.
    Returns a partial-state dict that phase 2 + on_generate_images will unpack.
    Includes narrator so it survives into the chained call.
    """
    log.info("run_text_generation: generating room JSON on CPU...")
    text_model = misc_f.load_text_model()
    room = misc_f.generate_game(text_model)
    log.info("run_text_generation: done — '%s'", room.get("room_name"))
    return {"room": room, "narrator": narrator}


@spaces.GPU(duration=60)
def run_image_generation(text_result: dict):
    """
    Phase 2: GPU-only image generation with FLUX.
    Receives the dict from run_text_generation so this frame starts
    immediately with inference.
    Returns (room, narrator, room_img, loc0-3, door_img, [key0,key1,door_key]).
    """
    room     = text_result["room"]
    narrator = text_result["narrator"]
    log.info("run_image_generation: loading FLUX...")
    image_model = misc_f.load_image_model()

    room_img = misc_f.generate_image(image_model, room["room_prompt"],                               "room")
    loc0     = misc_f.generate_image(image_model, room["containers"][0]["container_prompt"],         "location")
    loc1     = misc_f.generate_image(image_model, room["containers"][1]["container_prompt"],         "location")
    loc2     = misc_f.generate_image(image_model, room["containers"][2]["container_prompt"],         "location")
    loc3     = misc_f.generate_image(image_model, room["containers"][3]["container_prompt"],         "location")
    door_img = misc_f.generate_image(image_model, room["door_prompt"],                               "location")
    key0     = misc_f.generate_image(image_model, room["keys"][0]["key_prompt"],                     "item")
    key1     = misc_f.generate_image(image_model, room["keys"][1]["key_prompt"],                     "item")
    door_key = misc_f.generate_image(image_model, room["door_key_prompt"],                           "item")

    return room, narrator, room_img, loc0, loc1, loc2, loc3, door_img, [key0, key1, door_key]


# ─── Text-only inference (CPU, no GPU frame) ──────────────────────────────────

def run_continue(container: str, key: str, right_key: bool, item_given: str = "") -> str:
    """Narrate a container-opening attempt. CPU singleton — no ZeroGPU frame."""
    text_model = misc_f.load_text_model()
    result = misc_f.continue_game(text_model, container, key, right_key, item_given)
    return result.get("text", "")


def run_open_door(key: str, key_type: str) -> str:
    """Narrate a door-opening attempt. CPU singleton — no ZeroGPU frame."""
    text_model = misc_f.load_text_model()
    result = misc_f.open_door(text_model, key, key_type)
    return result.get("text", "")


# ─── TTS (GPU frame) ──────────────────────────────────────────────────────────

@spaces.GPU(duration=60)
def run_narration(narrator: str, text: str) -> str:
    """
    Synthesise speech for `text` using the chosen narrator voice.
    Always called via .then() after the narrative textbox is already updated,
    so the UI is never blocked waiting for audio.
    Returns a file path that gr.Audio can serve.
    """
    if not text or not narrator:
        return None
    log.info("run_narration: synthesising for narrator '%s'", narrator)
    tts_model = misc_f.load_tts_model()
    return misc_f.generate_voice(tts_model, text, narrator)


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="1000 Rooms") as demo:

    # ── Persistent state ──────────────────────────────────────────────────────
    # room_state dict:
    #   "room"         – raw room dict from generate_game()
    #   "chain"        – generate_random_chain() output
    #   "loc_imgs"     – list[4] PIL images for containers
    #   "key_imgs"     – list[3] PIL images [key0, key1, door_key]
    #   "found_keys"   – set of found key ids: "key0"|"key1"|"door_key"
    #   "tried_combos" – set of (box_idx, key_id_or___none__) already tried
    #   "door_opened"  – bool
    #   "narrator"     – chosen narrator name string

    room_state    = gr.State(None)
    sel_container = gr.State(-1)   # 0-3, -1 = nothing selected
    sel_key       = gr.State("")   # "key0"|"key1"|"door_key"|""

    # Intermediate state between the two generation phases
    _text_result  = gr.State(None)

    gr.HTML("""
    <div id="nn-overlay" class="hidden">
        <div id="nn-label">Generating…</div>
        <canvas id="nn-canvas"></canvas>
    </div>
    """)

    # ── Generate screen ────────────────────────────────────────────────────────
    with gr.Column(visible=True, elem_id="gen-screen") as gen_screen:
        gr.HTML("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:60vh;gap:2rem;">
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

        with gr.Column(elem_id="narrator-picker"):
            gr.HTML("""
            <div style="text-align:center;font-family:'Share Tech Mono',monospace;
                        font-size:0.95rem;color:#00ffcc;letter-spacing:0.12em;
                        margin-bottom:0.75rem;">
                Who is going to come with you?
            </div>
            """)
            narrator_radio = gr.Radio(
                choices=list(misc_f.VOICES.keys()),
                label="", show_label=False,
                interactive=True, value=list(misc_f.VOICES.keys())[0],
                elem_id="narrator-radio",
            )

        gen_btn    = gr.Button("Loading models…", variant="primary",
                               elem_id="gen-btn", interactive=False)
        status_lbl = gr.HTML(
            '<p style="text-align:center;color:#4a9f8f;font-size:0.75rem;'
            'font-family:\'Share Tech Mono\',monospace;margin-top:-0.5rem;">'
            'Downloading model weights, please wait…</p>',
            elem_id="status-lbl",
        )

    # ── Main screen ────────────────────────────────────────────────────────────
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
                    lines=15, elem_id="narrative-box", interactive=False,
                )
                narration_audio = gr.Audio(
                    label="", show_label=False,
                    autoplay=True, visible=False,
                    elem_id="narration-audio",
                )

        with gr.Row():
            with gr.Column(scale=3, min_width=160):
                gr.HTML('<div class="hud">The Door</div>')
                mid_img = gr.Image(
                    label="", show_label=False,
                    height=200, width=200,
                    interactive=False, elem_id="mid-img",
                )

            with gr.Column(scale=3, min_width=400):
                gr.HTML('<div class="hud">Inventory</div>')
                stack_gallery = gr.Gallery(
                    label="", show_label=False,
                    columns=3, rows=1, height=200,
                    object_fit="cover", allow_preview=False,
                    fit_columns=False, elem_id="stack-gallery", interactive=False,
                )

            with gr.Column(scale=3, min_width=120):
                gr.HTML('<div class="hud">The Key</div>')
                small_img = gr.Image(
                    label="", show_label=False,
                    height=200, width=200,
                    interactive=False, elem_id="small-img",
                )

            with gr.Column(scale=4, min_width=160):
                gr.HTML('<div class="hud">Selection</div>')
                selection_display = gr.Textbox(
                    label="", show_label=False,
                    value="Container: —\nKey: —",
                    lines=3, interactive=False,
                    elem_id="selection-display",
                )

        with gr.Row():
            cont_btn = gr.Button(
                "Continue →", variant="primary",
                scale=1, elem_id="cont-btn",
            )
            door_btn = gr.Button(
                "Open Door 🚪", variant="secondary",
                scale=1, elem_id="door-btn", visible=True,
            )

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_warmup_complete(_status):
        return (
            gr.update(value="Generate", interactive=True),
            gr.update(visible=False),
        )

    # ── Generation — two chained steps ────────────────────────────────────────
    # Step A (gen_btn.click → run_text_generation):
    #   CPU only. Triggers nn-overlay via JS. Stores result in _text_result.
    #   Returns nothing to the UI — overlay stays up.
    #
    # Step B (.then → on_generate_images):
    #   Calls run_image_generation (GPU frame). Populates the main screen.
    #   main_img.change fires → JS hides the overlay.
    #   Followed by a .then for TTS narration of the room story.

    def phase1_text(narrator: str) -> dict:
        """Step A: CPU text gen. Returns intermediate dict for step B."""
        return run_text_generation(narrator)

    def phase2_images(text_result: dict):
        """
        Step B: GPU image gen + state assembly.
        Returns all UI outputs needed to populate the main screen.
        """
        (room, narrator, room_img,
         loc0, loc1, loc2, loc3,
         door_img, key_imgs) = run_image_generation(text_result)

        loc_imgs = [loc0, loc1, loc2, loc3]
        chain    = generate_random_chain(
            list(range(4)),
            ["key0", "key1", "door_key"],
        )

        state = {
            "room":         room,
            "chain":        chain,
            "loc_imgs":     loc_imgs,
            "key_imgs":     key_imgs,
            "found_keys":   set(),
            "tried_combos": set(),
            "door_opened":  False,
            "narrator":     narrator,
        }

        story = room.get("room_story", "")
        return (
            gr.update(visible=False),   # gen_screen
            gr.update(visible=True),    # main_screen
            room_img,                   # main_img  ← triggers overlay hide
            loc_imgs,                   # gallery_top
            door_img,                   # mid_img
            [],                         # stack_gallery — empty
            None,                       # small_img — door key hidden
            story,                      # narrative
            state,                      # room_state
            -1,                         # sel_container
            "",                         # sel_key
            gr.update(visible=True),    # door_btn
            "Container: —\nKey: —",     # selection_display
        )

    def tts_for_room(state) -> tuple:
        """Step C: TTS narration of the room story. Fires after phase2_images."""
        if state is None:
            return gr.update(visible=False), None
        story    = state["room"].get("room_story", "")
        narrator = state.get("narrator", "Alfred")
        if not story:
            return gr.update(visible=False), None
        audio_path = run_narration(narrator, story)
        return gr.update(visible=True), audio_path

    # ── Gallery selection handlers ─────────────────────────────────────────────

    def on_container_select_full(evt: gr.SelectData, state, key_id):
        idx            = evt.index
        container_name = f"Container {idx}"
        key_name       = "—"
        if state is not None:
            room           = state["room"]
            container_name = room["containers"][idx].get("container_name", container_name)
            ki_map = {"key0": 0, "key1": 1, "door_key": 2}
            ki = ki_map.get(key_id)
            if ki is not None:
                key_name = (room["keys"][ki].get("key_name", key_id) if ki < 2
                            else room.get("door_key_name", "door key"))
        return idx, f"Container: {container_name}\nKey: {key_name}"

    def on_key_select(evt: gr.SelectData, state, container_idx):
        key_id    = ""
        key_name  = "—"
        cont_name = "—"
        if state is not None:
            found = sorted(state["found_keys"])
            if evt.index < len(found):
                key_id = found[evt.index]
            ki_map = {"key0": 0, "key1": 1, "door_key": 2}
            ki     = ki_map.get(key_id)
            if ki is not None:
                room     = state["room"]
                key_name = (room["keys"][ki].get("key_name", key_id) if ki < 2
                            else room.get("door_key_name", "door key"))
            if container_idx >= 0:
                cont_name = state["room"]["containers"][container_idx].get(
                    "container_name", f"Container {container_idx}"
                )
        return key_id, f"Container: {cont_name}\nKey: {key_name}"

    # ── Continue button ────────────────────────────────────────────────────────

    def on_continue(narrative_text, state, container_idx, key_id):
        """
        Validate → check tried_combos → run_continue() → reveal found key.
        When door is already opened, resets to the start screen.
        Returns narrative + state updates.  TTS fires in a chained .then().
        """
        if state is None:
            return (narrative_text, state,
                    gr.update(), gr.update(),
                    gr.update(visible=True), gr.update(visible=True))

        # Door opened → reset to start
        if state.get("door_opened"):
            return ("", None,
                    gr.update(), gr.update(),
                    gr.update(visible=True), gr.update(visible=False))

        room  = state["room"]
        chain = state["chain"]

        if container_idx < 0:
            msg = (narrative_text + "\n[Select a container first.]").strip()
            return (msg, state, gr.update(), gr.update(),
                    gr.update(visible=False), gr.update(visible=True))

        combo = (container_idx, key_id if key_id else "__none__")
        if combo in state["tried_combos"]:
            msg = (narrative_text + "\n[You've already tried that.]").strip()
            return (msg, state, gr.update(), gr.update(),
                    gr.update(visible=False), gr.update(visible=True))

        chain_entry = next((e for e in chain if e["box"] == container_idx), None)
        if chain_entry is None:
            return (narrative_text, state, gr.update(), gr.update(),
                    gr.update(visible=False), gr.update(visible=True))

        container_name = room["containers"][container_idx].get(
            "container_name", f"Container {container_idx}"
        )
        required_key = chain_entry["opened_by"]
        right_key    = (required_key is None and not key_id) or (required_key == key_id)

        item_given = ""
        if key_id:
            ki_map = {"key0": 0, "key1": 1, "door_key": 2}
            ki = ki_map.get(key_id)
            if ki is not None:
                item_given = (room["keys"][ki].get("key_name", key_id) if ki < 2
                              else room.get("door_key_name", key_id))

        narration_text = run_continue(container_name, key_id or "nothing", right_key, item_given)

        new_tried = set(state["tried_combos"]) | {combo}
        new_found = set(state["found_keys"])

        new_stack_gallery = gr.update()
        new_small_img     = gr.update()

        if right_key and chain_entry["contains"] is not None:
            found_key_id = chain_entry["contains"]
            if found_key_id not in new_found:
                new_found.add(found_key_id)
                ki_map = {"key0": 0, "key1": 1, "door_key": 2}
                if found_key_id == "door_key":
                    new_small_img = gr.update(value=state["key_imgs"][2])
                else:
                    inv_imgs = [
                        state["key_imgs"][ki_map[k]]
                        for k in sorted(new_found)
                        if k in ki_map and k != "door_key"
                    ]
                    new_stack_gallery = gr.update(value=inv_imgs)

        new_state        = {**state, "tried_combos": new_tried, "found_keys": new_found}
        updated_narrative = (narrative_text + "\n\n" + narration_text).strip()

        return (
            updated_narrative, new_state,
            new_stack_gallery, new_small_img,
            gr.update(visible=False), gr.update(visible=True),
        )

    def tts_for_continue(narrative_text, state) -> tuple:
        """TTS for the latest narration paragraph. Chained after on_continue."""
        if state is None or not narrative_text:
            return gr.update(visible=False), None
        # Narrate only the last paragraph (the new addition)
        last_para = narrative_text.strip().rsplit("\n\n", 1)[-1].strip()
        if not last_para or last_para.startswith("["):
            return gr.update(visible=False), None
        narrator   = state.get("narrator", "Alfred")
        audio_path = run_narration(narrator, last_para)
        return gr.update(visible=True), audio_path

    # ── Open door button ───────────────────────────────────────────────────────

    def on_open_door(state):
        """
        Checks possession of door_key. Narrates outcome.
        On success: hides door_btn, sets door_opened=True.
        TTS fires in a chained .then().
        """
        if state is None:
            return ("", state, gr.update(),
                    gr.update(visible=False), gr.update(visible=True))

        room  = state["room"]
        found = state["found_keys"]

        if "door_key" in found:
            key_type = "right"
            key_name = room.get("door_key_name", "the door key")
        elif found:
            key_type = "wrong"
            key_id   = sorted(found)[0]
            ki_map   = {"key0": 0, "key1": 1}
            ki       = ki_map.get(key_id, 0)
            key_name = (room["keys"][ki].get("key_name", key_id)
                        if ki < 2 else room.get("door_key_name", key_id))
        else:
            key_type = "none"
            key_name = "nothing"

        narration_text = run_open_door(key_name, key_type)

        if key_type == "right":
            new_state = {**state, "door_opened": True}
            return (narration_text, new_state,
                    gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=True))

        return (narration_text, state,
                gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=True))

    def tts_for_door(narration_text, state) -> tuple:
        """TTS for door narration. Chained after on_open_door."""
        if not narration_text or state is None:
            return gr.update(visible=False), None
        narrator   = state.get("narrator", "Alfred")
        audio_path = run_narration(narrator, narration_text)
        return gr.update(visible=True), audio_path

    # ── Warmup wiring ─────────────────────────────────────────────────────────
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

    # ── Generate wiring — three chained steps ─────────────────────────────────
    # click  → phase1_text   (CPU, overlay shown via JS)
    # .then  → phase2_images (GPU, overlay hidden when main_img updates)
    # .then  → tts_for_room  (GPU, narrate room story)

    gen_btn.click(
        fn=phase1_text,
        inputs=[narrator_radio],
        outputs=[_text_result],
        js="""() => {
            const ov = document.getElementById('nn-overlay');
            if (ov) ov.classList.remove('hidden');
            startNN();
            return [];
        }""",
    ).then(
        fn=phase2_images,
        inputs=[_text_result],
        outputs=[
            gen_screen, main_screen,
            main_img, gallery_top, mid_img,
            stack_gallery, small_img,
            narrative, room_state,
            sel_container, sel_key,
            door_btn, selection_display,
        ],
    ).then(
        fn=tts_for_room,
        inputs=[room_state],
        outputs=[narration_audio, narration_audio],
    )

    main_img.change(
        fn=None, inputs=[], outputs=[],
        js="""() => {
            const ov = document.getElementById('nn-overlay');
            if (ov) ov.classList.add('hidden');
            stopNN();
        }""",
    )

    # ── Gallery selection wiring ──────────────────────────────────────────────
    gallery_top.select(
        fn=on_container_select_full,
        inputs=[room_state, sel_key],
        outputs=[sel_container, selection_display],
    )

    stack_gallery.select(
        fn=on_key_select,
        inputs=[room_state, sel_container],
        outputs=[sel_key, selection_display],
    )

    # ── Continue wiring ───────────────────────────────────────────────────────
    cont_btn.click(
        fn=on_continue,
        inputs=[narrative, room_state, sel_container, sel_key],
        outputs=[narrative, room_state, stack_gallery, small_img, gen_screen, main_screen],
    ).then(
        fn=tts_for_continue,
        inputs=[narrative, room_state],
        outputs=[narration_audio, narration_audio],
    )

    # ── Door wiring ───────────────────────────────────────────────────────────
    door_btn.click(
        fn=on_open_door,
        inputs=[room_state],
        outputs=[narrative, room_state, door_btn, gen_screen, main_screen],
    ).then(
        fn=tts_for_door,
        inputs=[narrative, room_state],
        outputs=[narration_audio, narration_audio],
    )

    # ── Gallery selection highlight JS ────────────────────────────────────────
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