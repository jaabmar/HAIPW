import numpy as np
from anthropic import Anthropic
from openai import OpenAI
from HAIPW.faheyS78.utils_data import (
    TREATMENT_CONDITION, CONTROL_CONDITION, PROMPTS, QUESTION, VALUE_MAPPINGS,
    CLAUDE_KEY, GPT_KEY, DEEPSEEK_KEY,
    calculate_running_mse, extract_numeric_response
)
from HAIPW.utils import log


def get_client(model_name):
    clients = {
        "claude_haiku": Anthropic(api_key=CLAUDE_KEY),
        "deepseek": OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com"),
        "gpt4o": OpenAI(api_key=GPT_KEY)
    }
    return clients.get(model_name, clients["gpt4o"])


def get_claude_response(persona, user_prompt, anthropic_client):
    message = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=50,
        system=persona,
        temperature=1.0,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return message.content[0].text


def get_gpt_response(persona, user_prompt, openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": persona}, {"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content


def get_deepseek_response(persona, user_prompt, openai_client):
    response = openai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": persona}, {"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content


def get_llm_response(persona, user_prompt, model_name, client):
    model_dispatch = {
        "claude_haiku": get_claude_response,
        "deepseek": get_deepseek_response,
        "gpt4o": get_gpt_response
    }
    response = model_dispatch.get(model_name, get_gpt_response)(persona, user_prompt, client)
    numeric_val = extract_numeric_response(response)
    return numeric_val if numeric_val is not None and 1 <= numeric_val <= 5 else 3.0


def get_ensemble_prediction(persona, scenario_text, prompts, question, model_name, client):
    responses = [
        get_llm_response(persona, f"{scenario_text}\n\n{question}\n{p}", model_name, client)
        for p in prompts
    ]
    return responses


def generate_synthetic_data(df, model_name):
    client = get_client(model_name)
    mse_y1, mse_y0 = {i: [] for i in range(1, len(PROMPTS) + 1)}, {i: [] for i in range(1, len(PROMPTS) + 1)}
    df["Y1hat_responses"], df["Y0hat_responses"] = None, None
    df = df.astype({"Y1hat_responses": "object", "Y0hat_responses": "object"})
    for index, row in df.iterrows():
        persona = (
            f"You are a {row['AGE']} year old, {VALUE_MAPPINGS['PARTYID7'][int(row['PARTYID7'])]}, "
            f"gender {VALUE_MAPPINGS['GENDER'][int(row['GENDER'])]}, and hold {VALUE_MAPPINGS['IDEO'][int(row['IDEO'])]} views. "
            f"Additionally, your religion is {VALUE_MAPPINGS['RELIG'][int(row['RELIG'])]} and you "
            f"{VALUE_MAPPINGS['ATTEND'][int(row['ATTEND'])]} attend religious services. "
            f"You reside in {VALUE_MAPPINGS['HOME_TYPE'][int(row['HOME_TYPE'])]}. "
            f"You are responding to a scenario reflecting a debate involving college campus events and broader social issues. "
            f"Your answer must be in JSON format with an integer, without additional text. "
        )

        y1_responses = get_ensemble_prediction(persona, TREATMENT_CONDITION, PROMPTS, QUESTION, model_name, client)
        y0_responses = get_ensemble_prediction(persona, CONTROL_CONDITION, PROMPTS, QUESTION, model_name, client)

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
