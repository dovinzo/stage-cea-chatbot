#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system(' pip install transformers trl accelerate torch bitsandbytes peft datasets huggingface_hub -qU')


# In[2]:


from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os, torch, wandb
from transformers import HfArgumentParser, pipeline, logging


# In[3]:


notebook_login()


# In[4]:


instruct_tune_dataset = load_dataset("mosaicml/instruct-v3")


# In[5]:


instruct_tune_dataset = instruct_tune_dataset.filter(lambda x: x["source"] == "dolly_hhrlhf")


# In[6]:


instruct_tune_dataset["train"] = instruct_tune_dataset["train"].select(range(800))
instruct_tune_dataset["test"] = instruct_tune_dataset["test"].select(range(200))


# In[7]:


def create_prompt(sample):
  bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = "Answer the question."
  input = sample["prompt"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
  response = sample["response"]
  eos_token = "</s>"

  full_prompt = ""
  full_prompt += bos_token
  full_prompt += "### Instruction:"
  full_prompt += "\n" + system_message
  full_prompt += "\n\n### Input:"
  full_prompt += "\n" + input
  full_prompt += "\n\n### Response:"
  full_prompt += "\n" + response
  full_prompt += eos_token

  return full_prompt


# In[ ]:


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


# In[10]:


model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# In[11]:


def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")


# In[14]:


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)


# In[16]:


model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


# In[15]:


args = TrainingArguments(
  output_dir = "mistral_instruct_generation",
  #num_train_epochs=5,
  max_steps = 10, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 1,
  warmup_steps = 0,
  logging_dir='./logs',
  logging_steps=2,
  save_strategy="epoch",
  #evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=2, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=2e-4,
  bf16=True,
  lr_scheduler_type='constant',
)


# In[16]:


trainer = SFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=256,
  tokenizer=tokenizer,
  packing=True,
  formatting_func=create_prompt,
  args=args,
  train_dataset=instruct_tune_dataset["train"],
  eval_dataset=instruct_tune_dataset["test"]
)


# In[17]:


trainer.train()


# In[ ]:


new_model = "../models/Mistral-7b-v2-finetune-TEST"


# In[ ]:


trainer.model.save_pretrained(new_model)


# In[ ]:


#import os; os.getcwd()


# In[ ]:


#!pip install wandb


# In[29]:


#!pip uninstall wandb


# In[ ]:


model.config.use_cache = True


# In[ ]:


model.eval()


# In[ ]:


prompt = "Can I find information about the code's approach to handling long-running tasks and background jobs?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=50)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])


# In[ ]:


prompt = "Can I find information about SALOME?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])


# In[ ]:


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

modelb = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

tokenizerb = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

tokenizerb.pad_token = tokenizer.eos_token
tokenizerb.padding_side = "right"


# In[ ]:


pipeb = pipeline(task="text-generation", model=modelb, tokenizer=tokenizerb, max_length=400)


# In[ ]:


prompt = "Can I find information about SALOME?"
result = pipeb(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])


# In[ ]:


result[0]


# In[ ]:


result


# In[ ]:


prompt = ["<s>[INST]Can I find information about SALOME?[INST]", "<s>[INST]Can I find information about CEA?[INST]"]


# In[ ]:


result = pipeb(prompt)


# In[ ]:


result

