import json
import re

TREATMENT_CONDITION = """
We are now going to ask you to imagine you have read about the following scenario,
describing a debate on a recent College Campus.

Antifa Denied Permit to Protest on Campus, Provoking Debate About "Cancel Culture"

Columbus, OH

A debate on the merits of free speech erupted recently when the student chapter of the
controversial far-left group Antifa attempted to obtain a permit to conduct a demonstration
on the main quad of Rutgers University in New Jersey.

Citing safety concerns, the president of the organization in charge of Registered Student
Organizations (RSOs) initially denied the organization the right to conduct their rally,
arguing that their presence would endanger college students.

They cited a recent incident in Berkeley, CA where three Antifa members and two bystanders
were injured by rocks thrown in an altercation between the group and counter protesters.

A member of the local Antifa group, Luke Vargas, is appealing the decision, arguing that
the permit denial represented "cancel culture run amok," and the University was simply
"afraid to hear the truth."

When asked to comment, the University Ombudsman's Office promised that a final decision
on whether the rally would be permitted would be made by this Thursday, three days before
the march is scheduled to take place on Sunday.
"""

CONTROL_CONDITION = """
We are now going to ask you to imagine you have read about the following scenario,
describing a debate on a recent College Campus.

Antifa Denied Permit to Protest on Campus

Columbus, OH

A debate on the merits of free speech erupted recently when the student chapter of the
controversial far-left group Antifa attempted to obtain a permit to conduct a demonstration
on the main quad of Rutgers University in New Jersey.

Citing safety concerns, the president of the organization in charge of Registered Student
Organizations (RSOs) initially denied the organization the right to conduct their rally,
arguing that their presence would endanger college students.

They cited a recent incident in Berkeley, CA where three Antifa members and two bystanders
were injured by rocks thrown in an altercation between the group and counter protesters.

A member of the local Antifa group, Luke Vargas, promised to bring an appeal to the desk
of the University President.

When asked to comment, the University Ombudsman's Office promised that a final decision
on whether the rally would be permitted would be made by this Thursday, three days before
the march is scheduled to take place on Sunday.
"""

PROMPTS = [
    "Based on your perspective and the situation, how would you assess this scenario?",
    "Taking into account the family's context, how would you evaluate the priority level?",
    "Given the background and details provided, what numerical response feels appropriate?",
    "Considering all relevant factors, where would you place this on the scale?",
    "Reflecting on the scenario and using your reasoning, what value would you assign?",
    "From your understanding of the situation, how would you quantify this priority?",
    "Given your insights and the context described, what would be your evaluation?",
    "With the provided details in mind, how would you rate this on the scale?",
    "Considering all the information and your perspective, what score seems suitable?",
    "If you were to evaluate the priority here, what number aligns with your reasoning?",
]

VALUE_MAPPINGS = {
    "IDEO": {1: "Very liberal",
             2: "Somewhat liberal",
             3: "Moderate",
             4: "Somewhat conservative",
             5: "Very conservative",
             -1: "Unknown"},
    "GENDER": {0: "Unknown", 1: "Male", 2: "Female"},
    "RACETHNICITY": {1: "White, non-Hispanic",
                     2: "Black, non-Hispanic",
                     3: "Other, non-Hispanic",
                     4: "Hispanic",
                     5: "2+, non-Hispanic",
                     6: "Asian, non-Hispanic"},
    "HOME_TYPE": {
        1: "A one-family house detached from any other house",
        2: "A one-family house attached to one or more houses",
        3: "A building with 2 or more apartments",
        4: "A mobile home or trailer",
        5: "Boat, RV, van, etc",
    },
    "PARTYID7": {
        -1: "Unknown",
        1: "Strong Democrat",
        2: "Not so strong Democrat",
        3: "Lean Democrat",
        4: "Don't Lean/Independent/None",
        5: "Lean Republican",
        6: "Not so strong Republican",
        7: "Strong Republican",
    },
    "RELIG": {
        1: "Protestant",
        2: "Roman Catholic",
        3: "Mormon",
        4: "Orthodox",
        5: "Jewish",
        6: "Muslim",
        7: "Buddhist",
        8: "Hindu",
        9: "Atheist",
        10: "Agnostic",
        11: "Nothing in particular",
        12: "Just Christian",
        13: "Unitarian",
        14: "Something else",
        77: "DON'T KNOW",
        98: "SKIPPED ON WEB",
        99: "REFUSED",
    },
    "ATTEND": {
        1: "Never",
        2: "Less than once per year",
        3: "About once or twice a year",
        4: "Several times a year",
        5: "About once a month",
        6: "2-3 times a month",
        7: "Nearly every week",
        8: "Every week",
        9: "Several times a week",
        77: "DON'T KNOW",
        98: "SKIPPED ON WEB",
        99: "REFUSED",
    },
}

QUESTION = (
    "Generally speaking, do you agree or disagree with the following statement:\n"
    "Cancel culture is a big problem in today’s society. "
    "Choose an integer between 1 (definitely agree) and 5 (definitely disagree) to state your opinion."
)

CLAUDE_KEY = "claude_key"
GPT_KEY = "gpt_key"
DEEPSEEK_KEY = "deepseek_key"


def extract_numeric_response(response_string):
    """Extract a numeric response from structured or free text."""
    try:
        response_dict = json.loads(response_string)
        if isinstance(response_dict, dict):
            for value in response_dict.values():
                if isinstance(value, (int, float)):
                    return int(value)
        elif isinstance(response_dict, (int, float)):
            return int(response_dict)
    except (json.JSONDecodeError, KeyError):
        pass
    numbers = re.findall(r"\d+", response_string)
    if numbers:
        return int(numbers[0])
    return None


def calculate_running_mse(y_true, predictions_array):
    mse_scores = []
    for i in range(1, len(predictions_array) + 1):
        current_pred = sum(predictions_array[:i]) / i
        mse = (y_true - current_pred) ** 2
        mse_scores.append(mse)
    return mse_scores
