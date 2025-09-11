### Install Dependencies
```
pip install -r requirements.txt
```

### Finetuning DeepSeek-Coder-1.3B with LoRA
run the following command to finetune the DeepSeek-Coder-1.3B model using LoRA:
```
python finetune.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --output_dir ./results \
  --save_dir ./trained_model \
  --epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 512
```

### Generating Prompts with the Finetuned Model
```
python generate_prompts.py \
  --model_dir ./trained_model \
  --prompts_dir ./meus_prompts \
  --output_dir ./outputs \
  --k 5 \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --top_p 0.95 \
  --load_in_4bit

```