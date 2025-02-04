import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from HAIPW.faheyS78.utils_data import (
    TREATMENT_CONDITION, CONTROL_CONDITION, PROMPTS, QUESTION, VALUE_MAPPINGS,
    calculate_running_mse, extract_numeric_response
)
from HAIPW.utils import log

chat_pipeline = None


def load_pipeline(name):
    model_configs = {
        "llama": {
            "model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        },
        "llama_small": {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=False,
            )
        },
    }
    if name not in model_configs:
        raise ValueError(f"Invalid model name: {name}.")
    config = model_configs[name]
    model_name = config["model_name"]
    quantization_config = config["quantization_config"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    global chat_pipeline
    chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    chat_pipeline.tokenizer.pad_token_id = model.config.eos_token_id


def process_outputs(example):
    outputs = chat_pipeline(
        example["text"],
        max_new_tokens=100,
        eos_token_id=[
            chat_pipeline.tokenizer.eos_token_id,
            chat_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        do_sample=True,
        temperature=1.2,
        top_p=0.9,
        batch_size=len(example["text"]),
    )
    result_texts = []
    for prompt, output_item in zip(example["text"], outputs):
        response_str = output_item[0]["generated_text"]
        answer_str = response_str[len(prompt):].strip()
        numeric_val = extract_numeric_response(answer_str)
        if numeric_val is not None and 1 <= numeric_val <= 5:
            result_texts.append(numeric_val)
        else:
            result_texts.append(3.0)  # Middle value
    return {"responses": result_texts}


def get_template(persona, user_prompt):
    messages = [
        {"role": "system", "content": persona},
        {"role": "user", "content": user_prompt}
    ]

    return chat_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_ensemble_prediction(persona, scenario_text, prompts, question):
    batch_prompts = [get_template(persona, f"{scenario_text}\n\n{question}\n{p}") for p in prompts]
    ds = Dataset.from_dict({"text": batch_prompts})
    ds = ds.map(process_outputs, batched=True, batch_size=len(prompts))
    valid_responses = [r for r in ds["responses"] if r is not None]
    return (np.mean(valid_responses), valid_responses) if valid_responses else (None, [])


def generate_synthetic_data(df, model_name):
    load_pipeline(model_name)

    mse_y1, mse_y0 = {i: [] for i in range(1, len(PROMPTS) + 1)}, {i: [] for i in range(1, len(PROMPTS) + 1)}
    df["Y1hat_responses"] = None
    df["Y0hat_responses"] = None
    df = df.astype({"Y1hat_responses": "object", "Y0hat_responses": "object"})
    for index, row in df.iterrows():
        # Persona construction
        persona = (
            f"You are a {row['AGE']} year old, {VALUE_MAPPINGS['PARTYID7'][int(row['PARTYID7'])]}, "
            f"gender {VALUE_MAPPINGS['GENDER'][int(row['GENDER'])]}, and hold {VALUE_MAPPINGS['IDEO'][int(row['IDEO'])]} views. "
            f"Additionally, your religion is {VALUE_MAPPINGS['RELIG'][int(row['RELIG'])]} and you "
            f"{VALUE_MAPPINGS['ATTEND'][int(row['ATTEND'])]} attend religious services. "
            f"You reside in {VALUE_MAPPINGS['HOME_TYPE'][int(row['HOME_TYPE'])]}. "
            f"You are responding to a scenario reflecting a debate involving college campus events and broader social issues. "
            f"Your answer must be in JSON format with an integer, without additional text. "
        )

        # Treatment scenario with ensemble prompts
        _, y1_responses = get_ensemble_prediction(persona, TREATMENT_CONDITION, PROMPTS, QUESTION)
        # Control scenario with ensemble prompts
        _, y0_responses = get_ensemble_prediction(persona, CONTROL_CONDITION, PROMPTS, QUESTION)

        # Compute MSE
        if row['T'] == 1:
            mse_scores = calculate_running_mse(row['Y'], y1_responses)
            for i, score in enumerate(mse_scores, 1):
                mse_y1[i].append(score)
        else:
            mse_scores = calculate_running_mse(row['Y'], y0_responses)
            for i, score in enumerate(mse_scores, 1):
                mse_y0[i].append(score)

        # Calculate final predictions
        y1, y0 = np.mean(y1_responses), np.mean(y0_responses)

        # Store results
        df.at[index, "Y1hat"] = y1
        df.at[index, "Y0hat"] = y0
        df.at[index, "Y1hat_responses"] = y1_responses
        df.at[index, "Y0hat_responses"] = y0_responses

        log(f"Processed {index + 1}/{len(df)}")
        log(f"Y1 (treatment): {y1}, Responses: {y1_responses}")
        log(f"Y0 (control): {y0}, Responses: {y0_responses}")
        log(f"Y: {row['Y']}")
        log(f"T: {row['T']}")
        log("---")

    return df, mse_y1, mse_y0
