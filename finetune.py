import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

model_name = "deepseek-ai/deepseek-coder-1.3b-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_8bit=True
)

print("Model and tokenizer loaded successfully.")



# Load the dataset
# Load the hpcgroup/hpc-instruct dataset
dataset = load_dataset("hpcgroup/hpc-instruct")

# Filter the dataset for language 'Cuda'
cuda_dataset = dataset.filter(lambda example: example['language'] == 'Cuda')

# Use only a quarter of the cuda_dataset
quarter_cuda_dataset_size = len(cuda_dataset["train"]) // 4
quarter_cuda_dataset = cuda_dataset["train"].select(range(quarter_cuda_dataset_size))

# Define the formatting function
def format_example(example):
    # The hpcgroup/hpc-instruct dataset has 'instruction' and 'response' columns
    formatted_text = f"Instruction: {example['problem statement']}\nResponse: {example['solution']}"
    return {"text": formatted_text}

# Apply the formatting function to the filtered dataset
formatted_quarter_cuda_dataset = quarter_cuda_dataset.map(format_example)


# Tokenize the dataset
# We need to ensure the tokenizer is available from the previous step
def tokenize_function(examples):
    # Using padding='max_length' and truncation=True for consistency
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize the formatted dataset
tokenized_dataset = formatted_quarter_cuda_dataset.map(tokenize_function, batched=True)


print("Dataset loaded, filtered, formatted, and tokenized successfully using a quarter of the CUDA dataset.")
print(tokenized_dataset)


lora_config = LoraConfig(
    r=16,  # LoRA attention dimension
    lora_alpha=32,  # scaling factor
    lora_dropout=0.05, # dropout probability
    bias="none", # type of bias to use
    task_type="CAUSAL_LM",  # set to "CAUSAL_LM" for language modeling
)

print("QLoRa configuration created successfully.")
print(lora_config)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_steps=10,
    save_total_limit=3,
    fp16=True,
    push_to_hub=False,
    report_to="none",
)

print("Training arguments configured successfully.")
print(training_args)


trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=lora_config,
    args=training_args,
)
trainer.train()


