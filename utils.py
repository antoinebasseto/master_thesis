"""
This file contains utility functions that are used across the app.
"""

import json
import re
import string
from typing import Any, Dict, List, Literal, Optional, Tuple

import streamlit as st
from openai.types.chat.chat_completion import ChatCompletion

from config import Group


def page_setup(page_title: str) -> None:
    """
    Setup the page with the given title.

    :param page_title: The title of the page.
    :return: None
    """
    st.set_page_config(
        page_title=page_title, layout="wide", initial_sidebar_state="collapsed"
    )


def save_widget(key: str, new_name: Optional[str] = None) -> None:
    """
    Save the data of an entry in the session state to the results dictionnary.

    :param widget_key: The key of the widget.
    :param new_name: The new name, if any, to use in the results dictionnary.
    :return: None
    """
    assert key in st.session_state, f"{key} cannot be found in session_state"
    assert "results" in st.session_state, "'results' cannot be found in session_state"

    if new_name is not None:
        st.session_state["results"][new_name] = st.session_state[key]
    else:
        st.session_state["results"][key] = st.session_state[key]


#######################################
# CASE
#######################################


def get_case_index() -> int:
    return st.session_state["case_index"]


def get_case_description(case_index: int) -> str:
    """
    Get the case description from the data folder.

    :param case_index: The index of the case.
    :return: The case description.
    """
    return open(f"data/case_{case_index}.txt", "r").read()


def parse_case_description(case_description: str, citations: list[str] | None) -> str:
    """
    Parse the case description by adding citations to the text.

    :param case_description: The case description.
    :param citations: The list of citations.
    :return: The parsed case description.
    """
    if citations is None:
        return case_description

    for citation in citations:
        citation_stripped_of_punctuation = citation.strip(string.punctuation)

        # Use re.sub for case-insensitive replacement
        case_description = re.sub(
            re.escape(citation_stripped_of_punctuation),
            f":red-background[{citation_stripped_of_punctuation} \
                [{citations.index(citation) + 1}]]",
            case_description,
            flags=re.IGNORECASE,
        )

    return case_description


def get_group() -> Group:
    return st.session_state["results"]["group"]


def get_hypotheses(hypotheses_table: dict) -> Tuple[List[str], List[str]]:
    """
    :param hypotheses_table: The table of hypotheses as given by streamlit.
    :return: A tuple containing the list of hypotheses and the selected hypotheses.
    """

    hypotheses = [h["hypothesis"] for h in hypotheses_table["added_rows"]]

    selected_hypotheses = hypotheses
    if len(hypotheses_table["added_rows"]) != 0:
        if "selected" in hypotheses_table["added_rows"][0]:
            selected_hypotheses = [
                h["hypothesis"]
                for h in hypotheses_table["added_rows"]
                if h["selected"] is True
            ]

    return hypotheses, selected_hypotheses


@st.cache_data
def parse_message(
    message: str, hypotheses: List[str], selected_hypotheses: List[str], group: Group
) -> Tuple[List[str], str]:
    """
    Parse the JSON message from the AI, notably by adding citations to the
    text.

    :param message: The JSON message from the AI.
    :param hypotheses: The hypotheses used in the AI prompt.
    :param group: The group of the user.
    :return: A tuple containing the list of citations and the parsed message.
    """

    message_dict = json.loads(message)

    parsed_message = ""
    citations = []

    if group is Group.HYPOTHESIS_DRIVEN:
        message_dict = message_dict[selected_hypotheses[0]]
        parsed_message += f"**Evidence for {selected_hypotheses[0]}**\n\n"
        for e in message_dict["evidence_for"]:
            parsed_message += f"- {e['claim']}"
            for c in e["citations"]:
                if c not in citations:
                    citations.append(c)
                parsed_message += f" :red-background[[{citations.index(c) + 1}]]"
            parsed_message += "\n\n"

        parsed_message += f"**Evidence against {selected_hypotheses[0]}**\n\n"
        for e in message_dict["evidence_against"]:
            parsed_message += f"- {e['claim']}"
            for c in e["citations"]:
                if c not in citations:
                    citations.append(c)
                parsed_message += f" :red-background[[{citations.index(c) + 1}]]"
            parsed_message += "\n\n"

    if group is Group.RECOMMENDATIONS_DRIVEN:
        parsed_message += f"Recommended lead diagnosis: \
            **{message_dict['lead_diagnosis']}**\n\n"

        parsed_rationale = re.sub(
            r"\[(\d+)\]",
            lambda x: f":red-background[[{x.group(1)}]]",
            message_dict["rationale"],
        )

        parsed_message += f"**Rationale:** \
            {parsed_rationale}\n\n"

        for c in message_dict["citations"]:
            if c not in citations:
                citations.append(c)

    return citations, parsed_message


#######################################
# OPENAI API
#######################################


def get_client():
    return st.session_state["client"]


def get_model():
    return st.session_state["results"]["model"]


@st.cache_data(show_spinner=False)
def get_chat_completion(prompt: str, json_schema: Dict[str, Any]) -> ChatCompletion:
    """
    :param prompt: The prompt to send to the AI.
    :param json_schema: The JSON schema to use for the response.
    :return: The chat completion object from OpenAI.
    """
    completion = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": json_schema},
        temperature=0,
    )
    return completion


def get_latest_message_content(completion: ChatCompletion) -> str | None:
    """
    :param completion: The chat completion object from OpenAI.
    :return: String of the latest message in the thread.
    """
    return completion.choices[0].message.content


def get_ai_prompt(
    group: Literal[Group.HYPOTHESIS_DRIVEN, Group.RECOMMENDATIONS_DRIVEN],
    case_description: str,
    hypotheses: list[str],
) -> str:
    if group is Group.HYPOTHESIS_DRIVEN:
        return f"""
You are a highly knowledgeable and helpful clinical assistant specializing in providing detailed and accurate evaluations of diagnostic hypotheses. Your primary role is to assist clinicians by thoroughly examining and interpreting clinical cases.

You will be provided with a case description and multiple diagnostic hypotheses for this case, generated by the clinician you are assisting.

Your task is to:

    1. Provide both evidence supporting and contradicting the given hypothesis.
    2. Be meticulous in evaluating every piece of evidence available in the case description.
    3. Ensure each piece of evidence is assessed correctly, offering a balanced view of its implications.
    4. For every claim you make, provide a clear explanation as to why it supports or refutes the hypothesis. Your explanations should be detailed and understandable, intended to assist the clinician in making an informed decision.
    5. Quote direct passages from the case description to support your claims.

Guidelines for completing your task:

    - Direct Citations: Any time you use information from the case description, annotate the claim with direct, continuous passages from the text. If multiple passages are needed, cite each one separately WITHOUT using ellipses. Make sure you only cite passages that can be found in the case description!
    - Thorough Analysis: Take your time to think through each piece of evidence step-by-step. Consider all aspects of the case description and the diagnostic hypothesis.
    - Detailed Explanations: Provide comprehensive explanations that elucidate why the evidence supports or contradicts the hypothesis. Use clear, clinical reasoning to explain your thought process.
    - Balanced Evaluation: Ensure that your analysis is balanced and impartial, presenting a fair evaluation of all evidence.

Remember, the goal is to assist the clinician by providing a clear, well-reasoned analysis of the evidence for and against the diagnostic hypothesis. Your thoroughness and attention to detail are crucial for ensuring the clinician has all the necessary information to make an informed decision.

Take your time to do the task correctly and think things through step by step.

Answer using a JSON format.

Case description:
{case_description}

Diagnostic hypotheses:
{"\n".join(hypotheses)}
""".strip()

    if group is Group.RECOMMENDATIONS_DRIVEN:
        return f"""
You are a highly knowledgeable and helpful clinical assistant specializing in providing detailed and accurate evaluations of diagnostic hypotheses. Your primary role is to assist clinicians by thoroughly examining and interpreting clinical cases.

You will be provided with a case description and a list of diagnostic hypotheses for this case, generated by the clinician you are assisting.

Your task is to:

    1. Reflect upon the case description, evaluating each hypothesis independently to see if it presents a likely diagnosis for the given case. Be meticulous in evaluating every piece of evidence available in the case description. Ensure each piece of evidence is assessed correctly, offering a balanced view of its implications.
    2. Provide a lead diagnosis from the list of hypotheses provided. This diagnosis should be the most probable based on the evidence presented in the case description.
    3. Provide a rationale explaining why the lead diagnosis you chose is the most probable, citing the case description to support your claims.
    4. Your explanations should be detailed and understandable, intended to assist the clinician in making an informed decision.
    5. Quote direct passages from the case description to support your claims.

Here are the detailed guidelines for completing your task:
    - Thorough Analysis: Take your time to think through each piece of evidence step-by-step. Consider all aspects of the case description and the diagnostic hypotheses.
    - Direct Citations: Any time you use information from the case description, annotate the claim with direct, continuous passages from the text. If multiple passages are needed, cite each one separately WITHOUT using ellipses. Make sure you only cite passages that can be found in the case description! Citations will be collected in the "citations" list, and references to that list made in the "rationale" text, by using square brackets containing the index number of the corresponding citation, e.g. [1].
    - Detailed Explanations: Provide comprehensive explanations that elucidate why the evidence supports the chosen lead diagnosis. Use clear, clinical reasoning to explain your thought process.

Remember, the goal is to recommend the correct lead diagnosis. Your thoroughness and attention to detail are crucial for ensuring the clinician has all the necessary information to make an informed decision.

Take your time to do the task correctly and think things through step by step.

Answer using a JSON format.

Case description:
{case_description}

Diagnostic hypotheses:
{"\n".join(hypotheses)}
""".strip()


def get_json_schema(
    group: Literal[Group.RECOMMENDATIONS_DRIVEN, Group.HYPOTHESIS_DRIVEN],
    hypotheses: list[str],
) -> Dict[str, Any]:
    if group is Group.RECOMMENDATIONS_DRIVEN:
        return gen_json_schema_for_recommendation_driven()

    if group is Group.HYPOTHESIS_DRIVEN:
        return gen_json_schema_for_hypothesis_driven(hypotheses)


def gen_json_schema_for_recommendation_driven() -> Dict[str, Any]:
    return {
        "name": "get_diagnosis",
        "description": "Provide a diagnosis for the given case.",
        "schema": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "Your rationale for the lead_diagnosis you selected, citing the case description to support your claims.",
                },
                "citations": {
                    "type": "array",
                    "description": "Direct citation from the case description that back up the rationale",
                    "items": {"type": "string"},
                },
                "lead_diagnosis": {
                    "type": "string",
                    "description": "The diagnosis you believe is most likely based on the evidence presented, strictly chosen from the list of hypotheses provided.",
                },
            },
            "required": ["lead_diagnosis", "citations", "rationale"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def gen_json_schema_for_hypothesis_driven(options: List[str]) -> Dict[str, Any]:
    schema = {
        "name": "get_evidence_for_and_against",
        "description": "Provide evidence supporting and contradicting the given hypothesis.",
        "schema": {
            "type": "object",
            "properties": {},
            "required": [option for option in options],
            "additionalProperties": False,
        },
        "strict": True,
    }
    for option in options:
        schema["schema"]["properties"][option] = {
            "type": "object",
            "properties": {
                "evidence_for": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "A statement presenting evidence supporting the hypothesis, and explaining why.",
                            },
                            "citations": {
                                "type": "array",
                                "description": "Direct citation from the case description that back up the claim",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["claim", "citations"],
                        "additionalProperties": False,
                    },
                },
                "evidence_against": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "A statement presenting evidence refuting the hypothesis, and explaining why.",
                            },
                            "citations": {
                                "type": "array",
                                "description": "Direct citation from the case description that back up the claim",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["claim", "citations"],
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
            "required": ["evidence_for", "evidence_against"],
        }
    return schema
