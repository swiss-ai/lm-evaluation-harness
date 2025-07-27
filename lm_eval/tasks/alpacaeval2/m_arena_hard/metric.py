import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Generate base responses

BASE_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
BASE_TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
BASE_MODEL = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)


def generate_base_response(prompt):
    inputs = BASE_TOKENIZER(prompt, return_tensors="pt").to(BASE_MODEL.device)
    with torch.no_grad():
        outputs = BASE_MODEL.generate(**inputs, max_new_tokens=50, do_sample=False)
    decoded = BASE_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip()


# TODO: Save generated responses to a hf dataset


def load_base_responses(dataset_path="...", split="test"):
    dataset = load_dataset(dataset_path, split=split)
    return dataset["completion"]


# Compare base responses with generated responses

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
JUDGE_TOKENIZER = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)
JUDGE_MODEL = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)


def build_eval_prompt(prompt, completion, base_response, language):
    return f"""You are a helpful following assistant whose goal is to select the preferred (least wrong) output for a given instruction in {language}.

Which of the following answers is the best one for the given instruction in {language}. A good answer should follow these rules:
    1) It should be in {language},
    2) It should answer the request in the instruction,
    3) It should be factually correct and semantically comprehensible,
    4) It should be grammatically correct and fluent.

Instruction/Question: {prompt}
Answer (A): {completion}
Answer (B): {base_response}

FIRST provide a very concise comparison of the two answers in English. If one answer is better, briefly explain which you prefer and why. If both answers are identical or equally good or bad, briefly explain why.
SECOND, on a new line, state exactly one of 'Answer (A)' or 'Answer (B) or TIE' to indicate your choice of preferred response. Output this on a new line.

Your response should use the format: Comparison: <concise comparison and explanation>
Preferred: <'Answer (A)' or 'Answer (B)' or 'TIE'>
Strictly follow the mentioned format for your response. The last line must only contain the final answer. Do NOT output any other characters in the last line.
"""


def extract_binary_score(text):
    text = text.split("Preferred:")[1].strip()
    if text.startswith("Answer (A)"):
        return 1
    elif text.startswith("Answer (B)"):
        return 0
    elif text.startswith("TIE"):
        return 0.5
    else:
        print("Unknown response:", text)
        return 0


def evaluate_generation(doc, predictions):
    prompt = doc["prompt"]
    completion = predictions[0]
    base_response = load_base_responses()[doc["id"]]
    full_prompt = build_eval_prompt(prompt, completion, base_response)

    inputs = JUDGE_TOKENIZER(full_prompt, return_tensors="pt").to(JUDGE_MODEL.device)
    with torch.no_grad():
        outputs = JUDGE_MODEL.generate(**inputs, max_new_tokens=50, do_sample=False)
    decoded = JUDGE_TOKENIZER.decode(outputs[0], skip_special_tokens=True)

    model_response = decoded[len(full_prompt) :].strip()

    score = extract_binary_score(model_response)

    return {"score": score}
