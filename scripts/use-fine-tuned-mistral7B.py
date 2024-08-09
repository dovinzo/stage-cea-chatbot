import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

# - Taper la commande (via le Terminal): kill -15 [PID] ou Restart le kernel
# - Récupérer le modèle de base sur le Hub, le modèle fine-tuné correspond que aux adaptateurs QLoRA
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "models/mistral-salome-forum-finetune/checkpoint-5")

eval_prompt = "### Question: What is a good example of function?\n### Answer:"
eval_prompt = "### Question: Is SALOME a god?\n### Answer:"
eval_prompt = "### Question: What is CSF_DelayServerRequest?\n### Answer:"
eval_prompt = "### Question: What is GetNodeXYZ?\n### Answer:"
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))
