import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

# Set default device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

cwd = os.getcwd()
ckpt_path = os.path.join(cwd, "models", "checkpoints")  # Use os.path.join
model_name = next((f for f in os.listdir(ckpt_path) if f.endswith(('.ckpt', '.safetensors'))), None)
print(model_name)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" keyf

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

from nodes import (
    KSampler,
    VAEDecode,
    ImageScale,
    CheckpointLoaderSimple,
    EmptyLatentImage,
    CLIPTextEncode,
    NODE_CLASS_MAPPINGS,
    SaveImage,
)
#importing Everything from layer diffusion (may be too much idk)

from layered_diffusion import *

def main():
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name = model_name
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            #textgoeshere
            text="(monster:1.3), (beast:1.2), burning, fire monster, upper body, portrait, solo, looking at viewer",
            #textendshere
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="AissistV2-neg", clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        cliptextencode_37 = cliptextencode.encode(
            #Beetextstart
            text="volcanos, lava, background",
            #Beetextend
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_38 = cliptextencode.encode(
            #Ceestart
            text="lava, texture",
            #Ceeend
            clip=get_value_at_index(checkpointloadersimple_4, 1)
        )
        #Layer Diffuse Apply code goes here-----------------

        layerdiffusion = LayeredDiffusionFG()
        layereddiffusionfg_13 = layerdiffusion.apply_layered_diffusion(
            config = "SD15, Attention Injection, attn_sharing",
            weight = 1,
            model = get_value_at_index(checkpointloadersimple_4, 0)
        )


        ksampler = KSampler()
        vaedecode = VAEDecode()
        imagescale = ImageScale()
        saveimage = SaveImage()

        for q in range(1):
            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=7,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(layereddiffusionfg_13, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_14 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )
            #Layer Diffuse Decode code goes here

            layerdiffusionRGBA = LayeredDiffusionDecodeRGBA()
            layereddiffusionRGBA_36 = layerdiffusionRGBA.decode(
                samples=get_value_at_index(ksampler_3, 0),
                images=get_value_at_index(vaedecode_14, 0),
                sd_version = "SD15",
                sub_batch_size = 16,
            )

            imagescale_45 = imagescale.upscale(
                upscale_method="nearest-exact",
                width=768,
                height=768,
                crop="disabled",
                image=get_value_at_index(layereddiffusionRGBA_36, 0),
            )

            ksampler_39 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=7,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(cliptextencode_37, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            ksampler_40 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=7,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(cliptextencode_38, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_41 = vaedecode.decode(
                samples=get_value_at_index(ksampler_39, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            vaedecode_42 = vaedecode.decode(
                samples=get_value_at_index(ksampler_40, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            imagescale_46 = imagescale.upscale(
                upscale_method="nearest-exact",
                width=768,
                height=768,
                crop="disabled",
                image=get_value_at_index(vaedecode_41, 0),
            )
            saveimage_9 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(imagescale_46, 0)
            )
            saveimage_10 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(imagescale_45, 0)
            )
            saveimage_11 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_42, 0)
            )


if __name__ == "__main__":
    main()
