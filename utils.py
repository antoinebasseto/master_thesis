"""
This file contains utility functions that are used across the app.
"""

import json
import re
import string

import streamlit as st

from config import Group


def page_setup(page_title: str) -> None:
    """
    Setup the page with the given title. Optionally, show the debug sidebar.

    :param page_title: The title of the page.
    :param debug: Whether to show the debug sidebar.
    :return: None
    """
    st.set_page_config(
        page_title=page_title, layout="wide", initial_sidebar_state="collapsed"
    )


def save_widget(key: str, new_name: str | None = None) -> None:
    """
    Save the data of an entry in the session state to the results dictionnary.

    :param widget_key: The key of the widget.
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


def get_hypotheses(group: Group, hypotheses_table: dict) -> list[str]:
    """
    Return the hypotheses selected by the user.

    :param group: The group of the user.
    :param hypotheses_table: The table of hypotheses as given by streamlit.
    :return: The hypotheses selected by the user.
    """
    if group is Group.HYPOTHESIS_DRIVEN:
        hypotheses = [
            h["hypothesis"]
            for h in hypotheses_table["added_rows"]
            if h["selected"] is True
        ]

    if group is Group.RECOMMENDATIONS_DRIVEN:
        hypotheses = [h["hypothesis"] for h in hypotheses_table["added_rows"]]

    return hypotheses


@st.cache_data
def parse_message(
    message: str, hypotheses: list[str], group: Group
) -> tuple[list[str], str]:
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
        parsed_message += f"**Evidence for {hypotheses[0]}**\n\n"
        for e in message_dict["evidence_for"]:
            parsed_message += f"- {e['claim']}"
            for c in e["citations"]:
                if c not in citations:
                    citations.append(c)
                parsed_message += f" :red-background[[{citations.index(c) + 1}]]"
            parsed_message += "\n\n"

        parsed_message += f"**Evidence against {hypotheses[0]}**\n\n"
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


def get_ai_prompt(group: Group, case_description: str, hypotheses: list[str]) -> str:

    if group is Group.HYPOTHESIS_DRIVEN:
        return f"""
        <CASE_DESCRIPTION>{case_description}</CASE_DESCRIPTION>


        <SUSPECT>
        {hypotheses[0]}
        </SUSPECT>"""

    if group is Group.RECOMMENDATIONS_DRIVEN:
        return f"""
        <CASE_DESCRIPTION>{case_description}</CASE_DESCRIPTION>


        <SUSPECTS>
        {'\n'.join(hypotheses)}
        </SUSPECTS>"""

    return ""


#######################################
# OPENAI API
#######################################


def get_client():
    return st.session_state["client"]


def get_model():
    return st.session_state["results"]["model"]


def get_assistant_id():
    return st.session_state["assistant"].id


@st.cache_data(show_spinner=False)
def get_run_and_thread_id(prompt: str) -> tuple[str, str]:
    """
    This function gets or creates a new thread and run in the OpenAI API
    and returns the thread_id and run_id. Because it is cached, it will
    only create a new thread and run once per prompt, otherwise just returning
    the already existing corresponding ids.

    :param prompt: The prompt to send to the AI.
    :return: A tuple containing the thread_id and run_id.
    """
    run = get_client().beta.threads.create_and_run(
        assistant_id=get_assistant_id(),
        thread={
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        model=get_model(),
    )
    return (run.thread_id, run.id)


def get_run_status(thread_id: str, run_id: str) -> str:
    """
    Retrieve the runs status given its and the thread's id.

    :param thread_id: ID of the thread being run.
    :param run_id: ID of the run to get the status of.
    :return: String of the status of the run.
    """
    run = get_client().beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return run.status


def get_latest_message_content(thread_id: str) -> str:
    """
    Retrieve the content of the latest message in a thread.

    :param thread_id: ID of the thread to get the message from.
    :return: String of the latest message in the thread.
    """
    return (
        get_client()
        .beta.threads.messages.list(thread_id, limit=1, order="desc")
        .data[0]
        .content[0]
        .text.value
    )
