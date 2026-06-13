# gradio and spaces must be imported before misc_f on ZeroGPU
import spaces
import gradio as gr

# misc_f sets torch._dynamo.config.disable=True at import time
import misc_f

import logging
import random
from PIL import Image, ImageDraw

# Static failure messages — picked randomly, no model inference needed
_CONTAINER_FAIL = misc_f.CONTAINER_FAIL_MESSAGES
_DOOR_FAIL      = misc_f.DOOR_FAIL_MESSAGES

import custom_styles

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
#
# Fixed chain — containers are always assigned the same roles:
#   box 0: always open (opened_by=None), may contain key0 or nothing
#   box 1: locked,      opened_by=key0,  contains key1
#   box 2: locked,      opened_by=key1,  contains door_key
#   box 3: always open (opened_by=None), may contain key0 or nothing
#
# key0 is placed in either box 0 or box 3 at random; the other open box
# is a red herring (contains=None).  Containers 1 and 2 are always locked
# in order.

def generate_fixed_chain():
    """Build the deterministic container-key chain.
    Returns a list of 4 dicts with keys: box, opened_by, contains.
    """
    # Randomly decide which open box hides key0
    open_boxes = [0, 3]
    random.shuffle(open_boxes)
    key0_box, empty_open_box = open_boxes

    chain = [None] * 4
    chain[key0_box]    = {"box": key0_box,    "opened_by": None,      "contains": "key0"}
    chain[empty_open_box] = {"box": empty_open_box, "opened_by": None, "contains": None}
    chain[1]           = {"box": 1,            "opened_by": "key0",    "contains": "key1"}
    chain[2]           = {"box": 2,            "opened_by": "key1",    "contains": "door_key"}
    return chain


# ─── Warmup ───────────────────────────────────────────────────────────────────
# Pre-caches all three model weights to HF disk so the main generation frame
# avoids any network round-trips.

def warmup_models():
    """Download all model weights to disk before any GPU frame runs.
    No GPU lease is consumed here — pure network + disk I/O.
    VoxCPM's ModelScope sub-dependency is fetched directly via the
    ModelScope SDK rather than by instantiating the model (which would
    trigger a CUDA init and crash outside a GPU frame)."""
    log.info("Warmup: pre-caching all model weights to disk (CPU only)...")
    misc_f._ensure_text_weights()          # Q4_K_M GGUF only
    log.info("Warmup: text model weights on disk.")
    misc_f._ensure_image_weights()         # FLUX weights
    log.info("Warmup: FLUX on disk.")
    misc_f._ensure_tts_weights()           # VoxCPM2 weights
    log.info("Warmup: VoxCPM2 on disk. All weights ready.")
    return "ready"


# ─── Two-frame generation pipeline ───────────────────────────────────────────
#
# Two @spaces.GPU frames avoids the ZeroGPU 429 rate-limit that fires when
# three leases are requested back-to-back for the same session.
#
# Flow driven by .then() chains in the button wiring below:
#
#   gen_btn.click → run_text_gen        (@GPU 140 s)
#                     GGUF → room JSON + narrative texts → gen_state
#       .then     → on_text_done        (CPU)
#                     stashes story text; overlay stays up
#       .then     → run_images_and_tts  (@GPU 420 s)
#                     FLUX → 9 images, then VoxCPM → audio → combined_state
#       .then     → on_all_done         (CPU)
#                     flips screens, populates all widgets, hides overlay


# ── Frame 1: text ─────────────────────────────────────────────────────────────

@spaces.GPU(duration=360)  # text + images + TTS combined
def run_generation_pipeline(narrator: str):
    """
    Single GPU frame:
        GGUF -> room generation
        FLUX -> images
        VoxCPM -> audio
    """

    narrator = (narrator or "").strip()
    if narrator not in misc_f.VOICES:
        narrator = next(iter(misc_f.VOICES))

    # ------------------------------------------------------------------
    # TEXT
    # ------------------------------------------------------------------

    log.info("Loading text model...")
    llm = misc_f.load_text_model()

    room = misc_f.generate_game(llm)
    chain = generate_fixed_chain()
    narrations = misc_f.pregenerate_text(llm, room, chain)

    misc_f.unload_text_model(llm)

    # ------------------------------------------------------------------
    # IMAGES
    # ------------------------------------------------------------------

    log.info("Loading image model...")
    image_model = misc_f.load_image_model()

    room_img = misc_f.generate_image(
        image_model,
        room["room_prompt"],
        "room"
    )

    loc_imgs = [
        misc_f.generate_image(
            image_model,
            room["containers"][i]["container_prompt"],
            "location"
        )
        for i in range(4)
    ]

    door_img = misc_f.generate_image(
        image_model,
        room["door_prompt"],
        "location"
    )

    key0 = misc_f.generate_image(
        image_model,
        room["keys"][0]["key_prompt"],
        "item"
    )

    key1 = misc_f.generate_image(
        image_model,
        room["keys"][1]["key_prompt"],
        "item"
    )

    door_key = misc_f.generate_image(
        image_model,
        room["door_key_prompt"],
        "item"
    )

    misc_f.unload_image_model(image_model)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    log.info("Loading TTS model...")
    tts_model = misc_f.load_tts_model()

    audio_paths = misc_f.pregenerate_audio(
        tts_model,
        room,
        chain,
        narrations,
        narrator,
    )

    misc_f.unload_tts_model(tts_model)

    # ------------------------------------------------------------------
    # RETURN EVERYTHING
    # ------------------------------------------------------------------

    return {
        "room": room,
        "chain": chain,
        "narrator": narrator,
        "narrations": narrations,
        "images": {
            "room_img": room_img,
            "loc_imgs": loc_imgs,
            "door_img": door_img,
            "key_imgs": [key0, key1, door_key],
        },
        "audio_paths": audio_paths,
    }


def on_generation_done(result):
    room = result["room"]

    state = {
        "room": room,
        "chain": result["chain"],
        "loc_imgs": result["images"]["loc_imgs"],
        "key_imgs": result["images"]["key_imgs"],
        "found_keys": set(),
        "tried_combos": set(),
        "door_opened": False,
        "narrator": result["narrator"],
        "narrations": result["narrations"],
        "audio_paths": result["audio_paths"],
    }

    story = room.get("room_story", "")

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        story,
        result["images"]["room_img"],
        result["images"]["loc_imgs"],
        result["images"]["door_img"],
        [],
        None,
        state,
        -1,
        "",
        gr.update(visible=True),
        "Container: —\nKey: —",
        gr.update(
            value=result["audio_paths"].get("room_story"),
            visible=bool(result["audio_paths"].get("room_story"))
        ),
    )
# ─── Gallery selection handlers ───────────────────────────────────────────────

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


# ─── Continue button ──────────────────────────────────────────────────────────

def on_continue(narrative_text, state, container_idx, key_id):
    """
    Correct combo  → look up pre-generated narration + play pre-baked audio.
    Wrong/no-key   → pick a random static failure message, no audio.
    Door opened    → reset to start screen.
    """
    _no_audio = gr.update(value=None, visible=False)

    if state is None:
        return (narrative_text, state,
                gr.update(), gr.update(),
                gr.update(visible=True), gr.update(visible=True),
                _no_audio)

    # Door opened → reset
    if state.get("door_opened"):
        return ("", None,
                gr.update(), gr.update(),
                gr.update(visible=True), gr.update(visible=False),
                _no_audio)

    chain = state["chain"]

    if container_idx < 0:
        msg = (narrative_text + "\n[Select a container first.]").strip()
        return (msg, state, gr.update(), gr.update(),
                gr.update(visible=False), gr.update(visible=True),
                _no_audio)

    combo = (container_idx, key_id if key_id else "__none__")
    if combo in state["tried_combos"]:
        msg = (narrative_text + "\n[You've already tried that.]").strip()
        return (msg, state, gr.update(), gr.update(),
                gr.update(visible=False), gr.update(visible=True),
                _no_audio)

    chain_entry = next((e for e in chain if e is not None and e["box"] == container_idx), None)
    if chain_entry is None:
        return (narrative_text, state, gr.update(), gr.update(),
                gr.update(visible=False), gr.update(visible=True),
                _no_audio)

    required_key = chain_entry["opened_by"]
    right_key    = (required_key is None and not key_id) or (required_key == key_id)

    new_tried = set(state["tried_combos"]) | {combo}
    new_found = set(state["found_keys"])

    new_stack_gallery = gr.update()
    new_small_img     = gr.update()
    audio_update      = _no_audio

    if right_key:
        # ── Correct open: pre-generated narration + pre-baked audio ───────
        narration_text = state["narrations"]["container_narrations"].get(container_idx, "")
        if not narration_text:
            narration_text = "You open the container and look inside."

        if chain_entry["contains"] is not None:
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

        audio_path = state["audio_paths"].get(f"container_{container_idx}")
        if audio_path:
            audio_update = gr.update(value=audio_path, visible=True)

    else:
        # ── Wrong or no key: static message, no audio ─────────────────────
        narration_text = random.choice(_CONTAINER_FAIL)

    new_state         = {**state, "tried_combos": new_tried, "found_keys": new_found}
    updated_narrative = (narrative_text + "\n\n" + narration_text).strip()

    return (
        updated_narrative, new_state,
        new_stack_gallery, new_small_img,
        gr.update(visible=False), gr.update(visible=True),
        audio_update,
    )


# ─── Open door button ─────────────────────────────────────────────────────────

def on_open_door(state):
    """
    Check possession of door_key. Look up pre-generated narration.
    On success: look up pre-generated narration + play pre-baked audio.
    No key / wrong key: pick a random static failure message, no audio.
    """
    _no_audio = gr.update(value=None, visible=False)

    if state is None:
        return ("", state, gr.update(),
                gr.update(visible=False), gr.update(visible=True),
                _no_audio)

    found = state["found_keys"]

    if "door_key" in found:
        # ── Correct key: pre-generated narration + pre-baked audio ────────
        narration_text = state["narrations"]["door_right"]
        if not narration_text:
            narration_text = "The door swings open. You step through into the light."
        audio_path   = state["audio_paths"].get("door_right")
        audio_update = gr.update(value=audio_path, visible=True) if audio_path else _no_audio
        new_state    = {**state, "door_opened": True}
        return (narration_text, new_state,
                gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=True),
                audio_update)

    # ── No key or wrong key: static message, no audio ─────────────────────
    narration_text = random.choice(_DOOR_FAIL)
    return (narration_text, state,
            gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=True),
            _no_audio)


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="1000 Rooms") as demo:

    # ── Persistent state ──────────────────────────────────────────────────────
    room_state     = gr.State(None)
    gen_state      = gr.State(None)   # text-gen output (room/chain/narrations)
    combined_state = gr.State(None)   # merged images+audio output
    sel_container  = gr.State(-1)
    sel_key        = gr.State("")
    narrator_state = gr.State(list(misc_f.VOICES.keys())[0])  # persists radio pick

    gr.HTML("""
    <div id="nn-overlay" class="hidden">
        <div id="nn-label">Generating…</div>
        <canvas id="nn-canvas"></canvas>
    </div>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700&family=Cinzel:wght@400;600&family=IM+Fell+English:ital@0;1&display=swap" rel="stylesheet" />
    """)

    # ── Generate screen ────────────────────────────────────────────────────────
    with gr.Column(visible=True, elem_id="gen-screen") as gen_screen:
        gr.HTML("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:60vh;gap:2rem;">
            <div style="font-family:'Cinzel Decorative',sans-serif;font-size:6rem;
                        font-weight:900;color:#f0c870;letter-spacing: 0.04em;
                        text-shadow:0 0 30px rgba(212, 127, 8, 0.4);">
                1000 Rooms
            </div>
            
            <div style="font-family:'Cinzel Decorative',monospace;font-size:2rem;
                        color:#c47c1a;letter-spacing:0.1em;">
                An escape room of infinite chambers
            </div>
            
            <div style="font-family:'IM Fell English',sans-serif;font-size:1rem;
                        color:#c47c1a;letter-spacing:0.2em;">
                The generation takes a long time: ~ 2 minutes on ZeroGPU
            </div>
        </div>
        """)

        with gr.Column(elem_id="narrator-picker"):
            gr.HTML("""
            <div style="text-align:center;font-family:'IM Fell English',monospace;
                        font-size:0.95rem;color:#c47c1a;letter-spacing:0.12em;
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
            '<p style="text-align:center;color:#c47c1a;font-size:0.75rem;'
            'font-family:\'IM Fell English\',monospace;margin-top:-0.5rem;">'
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
                    columns=2, rows=1, height=200,
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

    # Persist narrator selection into state so js= on gen_btn.click cannot clobber it
    narrator_radio.change(
        fn=lambda v: v,
        inputs=[narrator_radio],
        outputs=[narrator_state],
    )

    # ── Generate wiring — two GPU frames chained with .then() ──────────────────
    #
    # Step 1  gen_btn.click  → run_text_gen       (@GPU 140 s) → gen_state
    # Step 2  .then          → on_text_done        (CPU)        → stash story text
    # Step 3  .then          → run_images_and_tts  (@GPU 420 s) → combined state
    # Step 4  .then          → on_all_done         (CPU)        → flip screens + populate all
    #
    # FLUX and VoxCPM share one GPU lease (step 3), avoiding the ZeroGPU 429
    # rate-limit that fires when 3 leases are requested in quick succession.
    # on_all_done hides the overlay via JS once everything is ready.

    gen_btn.click(
    fn=run_generation_pipeline,
    inputs=[narrator_state],
    outputs=[combined_state],
    js="""() => {
        const ov = document.getElementById('nn-overlay');
        if (ov) ov.classList.remove('hidden');
        startNN();
    }"""
).then(
    fn=on_generation_done,
    inputs=[combined_state],
    outputs=[
        gen_screen,
        main_screen,
        narrative,
        main_img,
        gallery_top,
        mid_img,
        stack_gallery,
        small_img,
        room_state,
        sel_container,
        sel_key,
        door_btn,
        selection_display,
        narration_audio,
    ],
    js="""() => {
        const ov = document.getElementById('nn-overlay');
        if (ov) ov.classList.add('hidden');
        stopNN();
    }"""
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
    # on_continue returns audio path directly — no .then() needed for TTS.
    # narration_audio has autoplay=True, so updating its value triggers playback.

    cont_btn.click(
        fn=on_continue,
        inputs=[narrative, room_state, sel_container, sel_key],
        outputs=[
            narrative, room_state,
            stack_gallery, small_img,
            gen_screen, main_screen,
            narration_audio,
        ],
    )

    # ── Door wiring ───────────────────────────────────────────────────────────
    door_btn.click(
        fn=on_open_door,
        inputs=[room_state],
        outputs=[
            narrative, room_state, door_btn,
            gen_screen, main_screen,
            narration_audio,
        ],
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
