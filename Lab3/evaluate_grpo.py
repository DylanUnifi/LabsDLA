import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./grpo_qwen_final"

print(f"Loading tokenizer and model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# A classic "Countdown" style math puzzle
q = "Return a string of numbers and basic arithmetic operators (+, -, *, /) that evaluates to 24. The allowed numbers are 2, 3, 8, 8. You can use any number multiple times or not at all. You can use parenthesis."

prompt_text = (
    "You are a logical reasoning assistant.\n"
    "First, think step-by-step and write your reasoning inside <think> and </think> tags.\n"
    "Then, provide your final numerical answer strictly inside <answer> and </answer> tags.\n\n"
    f"Question: {q}"
)

messages = [
    {"role": "system", "content": "You are a helpful mathematical assistant."},
    {"role": "user", "content": prompt_text}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nGenerating response (this might take a few seconds)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n" + "="*50)
print("🎯 QUESTION:")
print(q)
print("\n🧠 MODEL RESPONSE:")
print(response)
print("="*50 + "\n")
