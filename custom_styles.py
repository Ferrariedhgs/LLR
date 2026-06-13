# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --bg:      #2c2720;
    --bg2:     #3d3838;
    --accent:  #e8a832;
    --glow:    rgba(232,168,50,0.25);
    --border:  #d4b896;
    --text:    #c9b89e;
    --dim:     #c47c1a;
    --mono:    'IM Fell English', monospace;
    --hud:     'Cinzel Decorative';
}

/* ── Body & container ── */
body, .gradio-container { background: var(--bg) !important; }
gradio-app { background: var(--bg) !important; }
.gradio-container { min-height: 100vh; }

/* ── Sparks canvas – injected by JS, fixed behind everything ── */
#sparks-bg {
    position: fixed !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    pointer-events: none !important;
    z-index: 0 !important;
    display: block !important;
}

/* Lift all Gradio content above sparks */
gradio-app > div,
.gradio-container > .main,
.contain { position: relative; z-index: 1; }

/* ── Hide Gradio chrome ── */
footer, .footer, .built-with, .svelte-byatnx { display: none !important; }

/* ── Panels / boxes ── */
.gr-panel, .gr-box, [class*="block"], .form,
.gr-form, .wrap { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ── Text ── */
label, .label-wrap span, p, h1, h2, h3 {
    font-family: var(--mono) !important;
    color: var(--text) !important;
}

/* ── Textbox ── */
textarea, input[type=text] {
    background: #0d3838 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-radius: 3px !important;
    caret-color: var(--accent) !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 10px var(--glow) !important;
}
#narrative-box textarea {
    height: 400px !important;
    min-height: 400px !important;
    resize: none !important;
    background: var(--bg2);
}

/* ── Dropdown / listbox ── */
.svelte-select, select, [class*="dropdown"] {
    background: #071e1e !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}

/* ── Buttons ── */
button.primary, button[class*="primary"], .gr-button-primary {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--hud) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    transition: box-shadow 0.2s, background 0.2s !important;
    cursor: pointer !important;
}
button.primary:hover { background: rgba(0,255,204,0.07) !important; box-shadow: 0 0 16px var(--glow) !important; }

button.secondary, button[class*="secondary"], .gr-button-secondary {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--dim) !important;
    font-family: var(--hud) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
}

/* ── Images ── */
div[data-testid="image"], .image-container, .thumbnail-item img {
    background: #071e1e !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}
.upload-container { background: #071e1e !important; }

/* ── Gallery shared ── */
.gallery-item, .thumbnail-item {
    position: relative !important;
    border: 2px solid var(--border) !important;
    border-radius: 3px !important;
    cursor: pointer !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
.gallery-item:hover { border-color: rgba(0,255,204,0.5) !important; }
#top-gallery .selected-gallery,
#stack-gallery .selected-gallery {
    border-color: var(--accent) !important;
    box-shadow: 0 0 16px var(--glow), inset 0 0 0 2px rgba(0,255,204,0.55) !important;
}
#top-gallery .selected-gallery::after,
#stack-gallery .selected-gallery::after {
    content: "" !important;
    position: absolute !important;
    inset: 0 !important;
    border: 2px solid var(--accent) !important;
    background: rgba(0,255,204,0.12) !important;
    pointer-events: none !important;
}

/* ── Top gallery: 400×400, four 200×200 cells ── */
#top-gallery { height: 400px !important; overflow: hidden !important; }
#top-gallery .grid-wrap {
    height: 400px !important;
    overflow: hidden !important;
    padding: 0 !important;
    gap: 0 !important;
}
#top-gallery .gallery-item {
    width: 200px !important;
    height: 200px !important;
}
#top-gallery .thumbnail-item {
    width: 200px !important;
    height: 200px !important;
}
#top-gallery .thumbnail-item img {
    width: 200px !important;
    height: 200px !important;
    object-fit: cover !important;
    display: block !important;
}

/* ── Stack gallery: 400×200, two 200×200 cells ── */
#stack-gallery { height: 200px !important; overflow: hidden !important; }
#stack-gallery .grid-wrap {
    height: 200px !important;
    overflow: hidden !important;
    padding: 0 !important;
    gap: 0 !important;
}
#stack-gallery .gallery-item {
    width: 200px !important;
    height: 200px !important;
}
#stack-gallery .thumbnail-item {
    width: 200px !important;
    height: 200px !important;
}
#stack-gallery .thumbnail-item img {
    width: 200px !important;
    height: 200px !important;
    object-fit: cover !important;
    display: block !important;
}

/* ── CheckboxGroup as visible list ── */
#listbox {
    background: #071e1e !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 4px !important;
    height: 200px !important;
    overflow-y: auto !important;
}
#listbox .wrap {
    background: transparent !important;
    border: none !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
#listbox label {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
    padding: 4px 8px !important;
    cursor: pointer !important;
    border-radius: 2px !important;
}
#listbox label:hover { background: rgba(0,255,204,0.07) !important; }
#listbox input[type=checkbox] { accent-color: var(--accent) !important; }

.selection-display {
background: var(--bg2);
}

#narration-audio {
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
}

/* ── HUD label ── */
.hud { font-family: var(--hud) !important; font-size: 0.55rem; letter-spacing: 0.2em;
       color: var(--dim) !important; text-transform: uppercase; margin-bottom: 2px; }

/* ── Loading overlay ── */
#nn-overlay {
    position: fixed; inset: 0;
    background: #2c2720;
    z-index: 9999;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 2rem;
    transition: opacity 0.4s;
}
#nn-overlay.hidden { opacity: 0; pointer-events: none; }

#nn-label {
    font-family: var(--hud);
    font-size: 1rem;
    letter-spacing: 0.25em;
    color: var(--accent);
    animation: pulse-text 1.4s ease-in-out infinite;
}
@keyframes pulse-text { 0%,100%{opacity:1} 50%{opacity:0.25} }

#nn-canvas { display: block; }


/* ── HARD RESET spacing ── */
.gradio-container {
    padding: 0 !important;
}

/* remove row gaps */
.gr-row {
    gap: 0 !important;
    margin: 0 !important;
}

/* remove column padding */
.gr-column {
    padding: 0 !important;
    margin: 0 !important;
}

/* kill internal block spacing */
.gr-block, .gr-form, .form {
    margin: 0 !important;
    padding: 0 !important;
}

/* tighten gallery wrapper */
#top-gallery .grid-wrap,
#stack-gallery .grid-wrap {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ensure images don't introduce spacing */
#top-gallery img,
#stack-gallery img {
    display: block !important;
    margin: 0 !important;
}

gradio-app {
    position: relative !important;
    z-index: 1 !important;
}

/* background canvas MUST be behind everything */
#sparks-bg {
    position: fixed !important;
    inset: 0 !important;
    z-index: 0 !important;
}

/* everything UI above sparks */
.gradio-container,
.gradio-container > div {
    position: relative !important;
    z-index: 2 !important;
}

html, body {
    height: 100%;
    overflow: hidden;
}
"""

# ─── JS (head injection) ─────────────────────────────────────────────────────

HEAD_JS = """
<script>
// ════════════════════════════════════════════════════════
//  Sparks – inject canvas into body after DOM ready
// ════════════════════════════════════════════════════════
(function waitForBody() {
    if (!document.body) { setTimeout(waitForBody, 50); return; }

    const canvas = document.createElement('canvas');
    canvas.id = 'sparks-bg';
    document.documentElement.appendChild(canvas);  // append, not prepend – avoids ordering issues
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
        canvas.style.transform = "translateZ(0)";
        canvas.style.willChange = "transform";
    }
    window.addEventListener('resize', resize);
    resize();

    function randSpark() {
        return {
            x:     Math.random() * canvas.width,
            y:     canvas.height + 10 + Math.random() * 30,
            vx:    (Math.random() - 0.5) * 0.7,
            vy:    -(0.6 + Math.random() * 2.2),
            life:  0.7 + Math.random() * 0.3,
            decay: 0.003 + Math.random() * 0.006,
            r:     0.8 + Math.random() * 2.2,
            hue:   30 + Math.random() * 20,
        };
    }

    const sparks = Array.from({length: 90}, randSpark);

    let tick = 0;
    function loop() {
        tick++;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        sparks.forEach((s, i) => {
            s.x  += s.vx + Math.sin(tick * 0.015 + i * 0.4) * 0.35;
            s.y  += s.vy;
            s.life -= s.decay;
            if (s.life <= 0 || s.y < -15) { sparks[i] = randSpark(); return; }
            ctx.save();
            ctx.globalAlpha = s.life * 0.9;
            ctx.shadowBlur  = 8;
            ctx.shadowColor = `hsl(${s.hue},100%,65%)`;
            ctx.fillStyle   = `hsl(${s.hue},100%,72%)`;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r * s.life, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        });
        requestAnimationFrame(loop);
    }
    loop();
})();

// ════════════════════════════════════════════════════════
//  Neural-net loading canvas
// ════════════════════════════════════════════════════════
let _nnHandle = null;

function startNN() {
    const cv = document.getElementById('nn-canvas');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    cv.width  = 300;
    cv.height = 180;

    const LAYERS = [2, 3, 3, 2];
    const W = 300, H = 180, PAD = 40;
    const cols = LAYERS.length;

    const nodes = LAYERS.map((n, ci) => {
        const x = PAD + ci * ((W - PAD*2) / (cols - 1));
        return Array.from({length: n}, (_, ni) => ({
            x,
            y: H/2 + (ni - (n-1)/2) * 44,
        }));
    });

    let phase = 0, phaseT = 0, last = null;
    const DUR = 550;

    function drawNN(ts) {
        if (!last) last = ts;
        const dt = ts - last; last = ts;
        phaseT += dt / DUR;
        if (phaseT >= 1) { phaseT = 0; phase = (phase + 1) % cols; }

        ctx.clearRect(0, 0, W, H);

        // wires
        for (let li = 0; li < cols - 1; li++) {
            nodes[li].forEach(a => nodes[li+1].forEach(b => {
                ctx.strokeStyle = '#3a2e1a';
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
            }));
        }

        // nodes
        nodes.forEach((layer, li) => {
            layer.forEach(n => {
                let radius = 6, color = '#c47c1a', glow = 0;
                if (li === phase) {
                    const t = phaseT < 0.5 ? phaseT * 2 : (1 - phaseT) * 2;
                    const e = t*t*(3-2*t);
                    radius = 6 + e * 7;
                    color  = '#e8a832';
                    glow   = e * 20;
                } else if (li < phase) {
                    color = 'rgba(196,124,26,0.3)';
                }
                ctx.save();
                if (glow) { ctx.shadowBlur = glow; ctx.shadowColor = '#e8a832'; }
                ctx.beginPath(); ctx.arc(n.x, n.y, radius, 0, Math.PI*2);
                ctx.fillStyle = color; ctx.fill();
                ctx.strokeStyle = li === phase ? '#e8a832' : '#c47c1a';
                ctx.lineWidth = 1.5; ctx.stroke();
                ctx.restore();
            });
        });

        _nnHandle = requestAnimationFrame(drawNN);
    }
    if (_nnHandle) cancelAnimationFrame(_nnHandle);
    _nnHandle = requestAnimationFrame(drawNN);
}

function stopNN() {
    if (_nnHandle) { cancelAnimationFrame(_nnHandle); _nnHandle = null; }
}

// ════════════════════════════════════════════════════════
//  Gallery selection toggle (called from onclick below)
// ════════════════════════════════════════════════════════
function toggleGalSel(el) {
    el.classList.toggle('selected-gallery');
}
</script>
"""