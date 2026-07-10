import re
import ast
import torch
import reasoning_gym
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

print("Loading Reasoning Gym 'countdown' dataset...")
# We use a small dataset size for demonstration. In a real scenario, this would be larger or procedurally generated continuously.
raw_dataset = reasoning_gym.create_dataset('countdown', size=500, seed=42)

# Format the dataset for GRPOTrainer
def format_data(examples):
    prompts = []
    answers = []
    for q, a in zip(examples['question'], examples['answer']):
        # We instruct the model to use <think> tags for its reasoning
        prompt_text = (
            "You are a logical reasoning assistant.\n"
            "First, think step-by-step and write your reasoning inside <think> and </think> tags.\n"
            "Then, provide your final numerical answer strictly inside <answer> and </answer> tags.\n\n"
            f"Question: {q}"
        )
        # GRPOTrainer expects 'prompt' (which can be a list of messages) and 'answer'
        prompts.append([
            {"role": "system", "content": "You are a helpful mathematical assistant."},
            {"role": "user", "content": prompt_text}
        ])
        answers.append(str(a))
    return {"prompt": prompts, "answer": answers}

# Convert reasoning_gym dataset to HuggingFace dataset
hf_dataset = Dataset.from_list(list(raw_dataset))
train_dataset = hf_dataset.map(format_data, batched=True, remove_columns=hf_dataset.column_names)

# --- Define Reward Functions ---

def format_reward_func(completions, **kwargs):
    """Reward +0.5 if output contains well-formed <think> and <answer> tags."""
    rewards = []
    for completion in completions:
        # Extract the assistant's generated text
        content = completion[0]["content"] if isinstance(completion, list) else completion
        if "<think>" in content and "</think>" in content and "<answer>" in content and "</answer>" in content:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def effort_reward_func(completions, **kwargs):
    """Reward +0.001 per token inside <think> tags to encourage exploration."""

    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            thought = match.group(1)
            rewards.append(0.001 * len(thought.split()))
        else:
            rewards.append(0.0)
    return rewards

def check_correctness(pred, ans):
    """Check if the predicted math expression evaluates to the same value as the answer
    and uses the exact same numbers. Uses ast.literal_eval for safety."""
    try:
        target_val = float(ast.literal_eval(str(ans)))
        pred_val = float(ast.literal_eval(str(pred)))
        if abs(target_val - pred_val) > 1e-5:
            return False
        # Extract all numbers from both strings
        ans_nums = sorted(re.findall(r'\d+', str(ans)))
        pred_nums = sorted(re.findall(r'\d+', str(pred)))
        if ans_nums != pred_nums:
            return False
        return True
    except (ValueError, SyntaxError, TypeError):
        return False

def accuracy_reward_func(completions, answer, **kwargs):
    """Reward +1.0 if the math expression evaluates to the target and uses the exact input numbers."""

    rewards = []
    for comp, ans in zip(completions, answer):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        # Extract answer
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            pred = match.group(1).strip()
            if check_correctness(pred, ans):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def brevity_penalty_func(completions, answer, **kwargs):
    """Penalty of -0.001 per token length to encourage conciseness, ONLY applied if the answer is correct."""

    rewards = []
    for comp, ans in zip(completions, answer):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        # Check if the answer is correct
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        is_correct = False
        if match:
            pred = match.group(1).strip()
            is_correct = check_correctness(pred, ans)
                
        # Only penalize length if it successfully solved the math problem
        if is_correct:
            rewards.append(-0.001 * len(content.split()))
        else:
            rewards.append(0.0)
    return rewards

# --- Setup Model and Trainer ---

if __name__ == "__main__":
    import os

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading Model: {model_id}")

    # For a 1.5B model on 3x 48GB GPUs, we can load it in bfloat16 without quantization.
    # (If using a 7B, you might want to enable LoRA or quantization here)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = "./grpo_qwen_reasoning"
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,   # Number of prompts per GPU
        gradient_accumulation_steps=4,
        num_generations=4,               # Group size (N=4 completions per prompt)
        max_completion_length=512,       # Max tokens for the reasoning + answer
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        report_to="none" # Set to "wandb" if you have an account configured
    )

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward_func, format_reward_func, effort_reward_func, brevity_penalty_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO Training! (Press Ctrl+C to interrupt)")
    # Only resume if a checkpoint directory exists
    resume = os.path.isdir(output_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(output_dir)
    )
    trainer.train(resume_from_checkpoint=resume)

    print("Training Complete! Saving model...")
    trainer.save_model("./grpo_qwen_final")
    print("Model saved to ./grpo_qwen_final")
