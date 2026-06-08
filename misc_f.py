from llama_cpp import Llama
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import soundfile as sf
from voxcpm import VoxCPM
import numpy as np
import spaces


GENERATE_GAME_PROMPT=r"""You are a creative dungeon-master AI that generates escape-room content. Always respond with a single valid JSON object by completing this template: "{"room_name":"",
    "room_story":"<COMPLETE>",
    "room_prompt":"<COMPLETE>",
    "door_description":"<COMPLETE>",
    "door_prompt":"<COMPLETE>",
    "door_key_name":"<COMPLETE>",
    "door_key_prompt":"<COMPLETE>",
    "containers":[
        {
            "container_name":"<COMPLETE>",
            "container_prompt":"<COMPLETE>",
        
        },
        {
            "container_name":"<COMPLETE>",
            "container_prompt":"<COMPLETE>",
        
        },
        {
            "container_name":"<COMPLETE>",
            "container_prompt":"<COMPLETE>",
        
        },
        {
            "container_name":"<COMPLETE>",
            "container_prompt":"<COMPLETE>",
        
        },
    ],
    "keys":[
        {
            "key_name":"<COMPLETE>",
            "key_prompt":"<COMPLETE>",
        },
        {
            "key_name":"<COMPLETE>",
            "key_prompt":"<COMPLETE>",
        },
    ]
}"""

CONTINUE_GAME_PROMPT=r"""You are a creative dungeon-master AI that generates responses based on the player's atempt at opening containers using keys. If the player succedees, reveal the item_to_give inside. Always respond with a single valid JSON object by completing this template: {
    "text":"<COMPLETE>"
}"""

OPEN_DOOR_PROMPT=r"""You are a creative dungeon-master AI that generates responses based on the player's atempt at opening doors using keys. If the player has no keys pursue him into searching the room. If the player hs the wrong keys describe how the key fails to open the door. If the player has the right key describe how the door opens. Always respond with a single valid JSON object by completing this template: {
    "text":"<COMPLETE>"
}"""

IMAGE_TYPES={
    "room":1024,
    "location":512,
    "item":512
}

VOICES={
    "Alfred":"A deep, calm male voice with slow pacing, clear articulation, and a warm, authoritative tone suitable for documentaries and storytelling",
    "May":"A soft, gentle female voice with medium-low pitch, smooth delivery, and a soothing tone ideal for audiobooks and explanations",
    "Liam":"A young adult male voice with energetic delivery, slightly fast speech, bright tone, and expressive intonation for dynamic content",
    "Ava":"A neutral synthetic assistant-like female voice with steady pacing, minimal emotion, and crisp articulation resembling a digital AI",
    "Arthur":"An elderly male voice with gravelly texture, slow thoughtful pacing, and a wise, reflective tone",
    "Margaret":"An elderly female voice with warm, slightly breathy tone, gentle pacing, and nurturing delivery",
    "Chloe":"A natural conversational female voice with warm friendliness, expressive but subtle intonation, and casual pacing",
    "Marcus":"A documentary narrator voice with deep tone, authoritative pacing, and strong clarity for storytelling",
    "Sergei":"A deep adult male voice with a russian accent, slightly resonant and strong",
    "Tatiana":"An adult female voice with a soft but distinct russian accent",
    "Su":"A mystical, ethereal female voice inspired by east asian folklore spirits"
}


#loads models into memory and returns their handles
def load_models():
    text_model = Llama.from_pretrained(
        repo_id="build-small-hackathon/Nemotron-nano-4b-escape-room",
        filename="nemotron-room-lora-Q4_K_M.gguf",
        n_ctx=2048,
    )

    image_model = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=torch.bfloat16, device_map="cuda")

    tts_model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

    return text_model, image_model, tts_model


#generates a game and returns the json
def generate_game(model):
    #model.reset()
    response = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": f"{GENERATE_GAME_PROMPT}"
            },
            {
                "role": "user",
                "content": """{"task":"genearate_room"}"""
            }
        ]
    )

    return response["choices"][0]["message"]["content"]


#tries to open a container with a key
def continue_game(model,container,key,right_key:bool,item_given=""):
    #model.reset()
    response = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": f"{CONTINUE_GAME_PROMPT}"
            },
            {
                "role": "user",
                "content": f'{{"task":"continue_game", "location_name":"{container}", "key_name":"{key}", "fits_lock":{right_key}, "item_to_give":"{item_given}"}}'
            }
        ]
    )

    return response["choices"][0]["message"]["content"]


#tries to open the door
def open_door(model,key,key_type):
    #model.reset()
    response = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": f"{OPEN_DOOR_PROMPT}"
            },
            {
                "role": "user",
                "content": f'{{"task":"open_door","has_key":"{key_type}","given_key":"{key}"}}'
            }
        ]
    )

    return response["choices"][0]["message"]["content"]

#generate and return an image
def generate_image(model,prompt,type):
    return model(prompt=prompt, width=IMAGE_TYPES[type], height=IMAGE_TYPES[type]).images[0]

#generate the narrator's voice
def generate_voice(model,prompt,narrator):
    chunks = []
    for chunk in model.generate_streaming(text=f'({VOICES[narrator]}) {prompt}'):
        chunks.append(chunk)
    wav = np.concatenate(chunks)
    sf.write("streaming.wav", wav, model.tts_model.sample_rate)
