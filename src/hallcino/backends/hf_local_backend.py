from .base import Backend, GenerationResult


class HFLocalBackend(Backend):
    """Loads a model (e.g. Gemma 4 12B) locally via transformers. Requires a GPU
    with enough VRAM for the chosen precision; install the 'local' extra first:
    pip install -e ".[local]"
    """

    def __init__(self, model: str, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map="auto"
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return GenerationResult(text=text, model=self.model_name)
