![Icon](https://github.com/agreene5/Procedural-Purgatory/assets/172552759/bb1a1af7-fb63-4b56-8c75-94500cc7a43b)


# Gameplay: https://www.youtube.com/watch?v=SqcZAa5_naE

# Requirements:

## -4GB Vram with an Nvidia CUDA GPU (pretty much any modern nvidia GPU)

## -16GB Ram

## -Python 3.10.x (newer versions might work, but I used 3.10)

## -Microsoft Visual Studio Build Tools (get the ‘Desktop development with C++’ when installing)

## -FluidSynth (and chocolatey to download it)

## -CUDA 11.8 (newer versions might work, but I used 11.8)

    Open the game file path in cmd and run this command to install pytorch for cuda 11.8.

    python -m pip install torch==2.3.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    Then, run this command in the same directory to install the python dependencies

    pip install -r requirements.txt

## -Stable Diffusion 1.5 model (the one I used was aingdiffusion which you can get here: https://civitai.com/models/34553/aingdiffusion) 

    Put the 1.5 model in the Procedural_Purgatory/models/checkpoints folder.

## -RWKV-4-MIDI models, get both the 120M and 560M models here: https://huggingface.co/BlinkDL/rwkv-4-music/tree/main 

    Put both models directly into the root of the folder (the Procedural_Purgatory folder)


If the Generate Textures or Generate Music button turns green for a couple seconds and then turns off, the python script isn’t working. You can manually run the “GodotImageGenerator.py” or “GodotMusicGenerator.py” to see what the error was.

If there’s an issue you’re unable to resolve in the “GodotMusicGenerator.py” file, you can change the following lines to run it on your CPU instead of GPU (this will be slower though).

os.environ[“RWKV_CUDA_ON”] = ‘1’          change to         os.environ[“RWKV_CUDA_ON”] = ‘0’

model = RWKV(model=MODEL_FILE, strategy=’cuda fp16’)          change to       model = RWKV(model=MODEL_FILE, strategy=cpu  fp16’)  

# What the game uses:
## - Stable Diffusion 1.5 (for base images)
## - Layer Diffusion (for transparent head/torso image)
## - RWKV-4-Music (for music generation)
## - Aubio (to predict BPM)
## - Fluidsynth (to convert .mid to .wav)

# Generation Times (RTX 3050 ti Laptop):
(Keep in mind the music times can vary greatly because the RWKV models don't always use the full token length they're given)

## Generate Textures: ~0:45
## Generate Music: ~4:30
## Generate Music (Faster Music Model): ~1:45
## Generate Music (Shorter Song): ~2:20
## Generate Music (Faster Music Model and Shorter Song): ~1:15

