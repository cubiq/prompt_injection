# Prompt Injection Node for ComfyUI

This custom node for ComfyUI allows you to inject specific prompts at specific blocks of the Stable Diffusion UNet, providing fine-grained control over the generated image.

## Highly Experimental

The code is very basic, experimental and prossibly buggy. It's a very interesting proof of concept and I will expand it if anything good can be done with it. 

At the moment this is a fork of [DataCTE](https://github.com/DataCTE/prompt_injection)'s repository, I'm in contact with them and we'll evaluate a merge when the code is stable.

## Credits

This code is based on [DataCTE](https://github.com/DataCTE/prompt_injection), [Perturbed Attention](https://github.com/pamparamm/sd-perturbed-attention), [B-Lora](https://github.com/yardenfren1996/B-LoRA/) and my previous experiments with the [IPAdapter](https://github.com/cubiq/ComfyUI_IPAdapter_plus?tab=readme-ov-file) style/composition.