#Modified/simplified version of the node from: https://github.com/pamparamm/sd-perturbed-attention
#If you want the one with more options see the above repo.

#My modified one here is more basic but has less chances of breaking with ComfyUI updates.

import comfy.model_patcher
import comfy.samplers
import torch

def build_patch(patchedBlocks, weight=1.0, sigma_start=0.0, sigma_end=1.0):
    def prompt_injection_patch(n, context_attn1: torch.Tensor, value_attn1, extra_options):
        (block, block_index) = extra_options.get('block', (None,None))
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9
        
        batch_prompt = n.shape[0] // len(extra_options["cond_or_uncond"])

        if sigma <= sigma_start and sigma >= sigma_end:
            if (block and f'{block}:{block_index}' in patchedBlocks and patchedBlocks[f'{block}:{block_index}']):
                c = context_attn1[0].unsqueeze(0)
                b = patchedBlocks[f'{block}:{block_index}'][0][0].to(context_attn1.device)
                if c.dim() == 2:
                    c = c.unsqueeze(0)
                out = torch.stack((c, b))
                out = out.repeat(1, batch_prompt, 1, 1) * weight

                return n, out, out 

        return n, context_attn1, value_attn1
    return prompt_injection_patch

class PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "in4":  ("CONDITIONING",),
                "in5":  ("CONDITIONING",),
                "in7":  ("CONDITIONING",),
                "in8":  ("CONDITIONING",),
                "mid0": ("CONDITIONING",),
                "out0": ("CONDITIONING",),
                "out1": ("CONDITIONING",),
                "out2": ("CONDITIONING",),
                "out3": ("CONDITIONING",),
                "out4": ("CONDITIONING",),
                "out5": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, in4=None, in5=None, in7=None, in8=None, mid0=None, out0=None, out1=None, out2=None, out3=None, out4=None, out5=None, weight=1.0, start_at=0.0, end_at=1.0):
        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        if any((in4, in5, in7, in8, mid0, out0, out1, out2, out3, out4, out5)):
            patchedBlocks = {
                'input:4': in4,
                'input:5': in5,
                'input:7': in7,
                'input:8': in8,
                'middle:0': mid0,
                'output:0': out0,
                'output:1': out1,
                'output:2': out2,
                'output:3': out3,
                'output:4': out4,
                'output:5': out5,
            }
            m.set_model_attn2_patch(build_patch(patchedBlocks, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class SimplePromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "block": (("output", "middle", "input"),),
                "index": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, block, index, conditioning=None, weight=1.0, start_at=0.0, end_at=1.0):
        if not conditioning:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        m.set_model_attn2_patch(build_patch({f"{block}:{index}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class AdvancedPromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "locations": ("STRING", {"multiline": True, "default": "output:0,1.0\noutput:1,1.0"}),
                "conditioning": ("CONDITIONING",),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, locations: str, conditioning=None, start_at=0.0, end_at=1.0):
        if not conditioning:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        for line in locations.splitlines():
            line = line.strip().strip('\n')
            weight = 1.0
            if ',' in line:
                line, weight = line.split(',')
                line = line.strip()
                weight = float(weight)
            if line:
                m.set_model_attn2_patch(build_patch({f"{line}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

NODE_CLASS_MAPPINGS = {
    "PromptInjection": PromptInjection,
    "SimplePromptInjection": SimplePromptInjection,
    "AdvancedPromptInjection": AdvancedPromptInjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptInjection": "Inject Prompt in Attention",
    "SimplePromptInjection": "Inject Prompt in Attention (simple)",
    "AdvancedPromptInjection": "Inject Prompt in Attention (advanced)"
}