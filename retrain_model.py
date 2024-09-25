from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

base_model = "/Users/gorbarseghyan/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b/2"
dataset_name = "ruslanmv/ai-medical-chatbot"
new_model = "/Users/gorbarseghyan/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b/2/llama-3-8b-chat-doctor"


torch.backends.mps.memory_debug_mode = True  # Enable memory debugging to track allocations
# Importing the dataset
dataset = load_dataset(dataset_name, split="all")
print(dataset)
dataset = dataset.shuffle(seed=65).select(range(1000))  # Only use 1000 samples for quick demo

# Set to float32 for CPU
torch_dtype = torch.float32
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32  # Use float32 for CPU
)


# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model, trust_remote_code=True, device_map='cpu'
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)


# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)


def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Patient"]},
                {"role": "assistant", "content": row["Doctor"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

print(dataset['text'][3])

dataset = dataset.train_test_split(test_size=0.1)

training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,  # You already have this low
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  # Lower accumulation to reduce memory pressure
    num_train_epochs=1,
    max_steps=100,  # If it's just a demo, reduce steps
    evaluation_strategy="steps",
    eval_steps=100,  # Adjust evaluation steps to reduce evaluation frequency
    logging_steps=10,  # Reduce logging frequency to reduce overhead
    learning_rate=2e-4,
    fp16=False,  # You are on CPU, so keep fp16 False
    bf16=False,  # Keep bf16 False as well
    group_by_length=True,
    report_to="wandb"
)



trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=256,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)


trainer.train()

