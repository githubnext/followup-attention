"""This downloads key stats about the users performances."""
from typing import Dict, List, Tuple, Union, Any
import re

from pymongo import mongo_client
from termcolor import colored
import pandas as pd

from attwizard.script.utils_mongodb import load_all_aggregated_records


def remove_user_from_delivery(nickname: str):
    """Remove user from delivery."""
    pass


def get_time_per_task(
        task_record: Dict[str, Any]) -> int:
    """Return the task duration in seconds."""
    start_time = int(task_record["openedPageTime"])
    if len(task_record["time"]) > 1:
        end_time = task_record["time"][-1]
    else:
        end_time = int(task_record["time"][0])
    task_duration = (end_time - start_time) / 1000
    return task_duration


def get_questions_and_answers(
        task_record: Dict[str, Any]) -> Dict[str, str]:
    """Extract the questions and answers from the given task.
    {
        "question": "What does the code print?",
        "answer": "Helloworld"
    }
    """
    try:
        answer_line = str(task_record['editableLineContent'])
        question = re.search(
            "Question(s)?:((?s:.*?))" + re.escape(answer_line),
            task_record["formattedcode"], re.MULTILINE).group(0)
    except Exception as e:
        print(task_record["formattedcode"][-200:])
        raise e
    # remove comments
    question = question.replace(answer_line, "").replace("#", "")
    question = question[question.index(":") + 1:]
    # remove multiple spaces or new lines
    question = re.sub(r"\s+", " ", question)
    answer = task_record["editableLineContent"]
    answer = answer.replace("Answer:", "")
    answer = answer.replace("#", "")
    answer = re.sub(r"\s+", " ", answer)
    return {
        "question": question,
        "answer": answer,
        "raw_answer": task_record["editableLineContent"]
    }


class PrintableLog(object):

    def __init__(self, file):
        self.file = file
        self.cumulative_content = ""

    def print(self, message, color=None):
        """print the log and store the new message."""
        if color:
            print(colored(message, color))
        else:
            print(message)
        self.cumulative_content += message + "\n"

    def dump(self):
        """Write the content to file."""
        with open(self.file, "w") as f:
            f.write(self.cumulative_content)


def get_tasks_nickname_with_score(
        nickname: str,
        all_data: List[Dict[str, Any]],
        df_google_form: pd.DataFrame,) -> List[Dict[str, Any]]:
    """Extract the records referring to the current user.

    Note that this will augment the records with the following fields:
    - task_duration_seconds (int)
    - task_duration_minutes (int)
    - question: (str)
    - answer: (str)
    - raw_answer: (str)
    - n_unblur_events: (int)
    - survey_score: (int)
     """
    user_tasks = [
        task for task in all_data if task["nickname"] == nickname]
    new_records = []
    # filter current user performance
    df_nickname = df_google_form[df_google_form["WorkerId"] == nickname]
    survey_score = df_nickname['Score'].values[0].split("/")[0] \
        if len(df_nickname) > 0 else 0
    for task in user_tasks:
        record = task
        time = get_time_per_task(task)
        time_in_minutes = time / 60
        record["task_duration_seconds"] = time
        record["task_duration_minutes"] = time_in_minutes
        record["n_unblur_events"] = len(task["hoverUnblurEvents"])
        q_and_a_dict = get_questions_and_answers(task)
        record["question"] = q_and_a_dict["question"]
        record["answer"] = q_and_a_dict["answer"]
        record["raw_answer"] = q_and_a_dict["raw_answer"]
        record["survey_score"] = survey_score
        record["completed"] = "experiment-submit" in task["flushReason"]
        new_records.append(record)
    return new_records


def txt_report_nickname(
        nickname: str,
        all_data: List[Dict[str, Any]],
        df_google_form: pd.DataFrame,
        output_filepath: str = None):
    """Create a textual report of the performance of the given nickname."""
    user_tasks = get_tasks_nickname_with_score(
        nickname, all_data, df_google_form)
    print("=" * 80)
    prLog = PrintableLog(output_filepath)
    prLog.print(f"Report of {nickname}")
    # get the corresponding nickname in the df_google_form
    df_nickname = df_google_form[df_google_form["WorkerId"] == nickname]
    if len(df_nickname) == 0:
        prLog.print(f"No google form found", color="red")
    completed_tasks = [
        task for task in user_tasks
        if "experiment-submit" in task["flushReason"]]
    prLog.print(f"Number of seen tasks: {len(user_tasks)}")
    prLog.print(f"{len(completed_tasks)} completed tasks")
    used_platforms = [
        task["platformname"] for task in user_tasks]
    prLog.print(f"Used platforms: {used_platforms}")
    all_task_times = []
    all_task_n_events = []
    for task in user_tasks:
        prLog.print("-" * 80)
        prLog.print(f"{task['sourceFile']} - {task['flushReason'][-1]}")
        time = get_time_per_task(task)
        time_in_minutes = time / 60
        all_task_times.append(time)
        n_unblur_events = len(task["hoverUnblurEvents"])
        all_task_n_events.append(n_unblur_events)
        color_time = "red" if (time < 60 or n_unblur_events < 50) else "green"
        prLog.print(
            f"Time: {time:.0f} sec ({time_in_minutes:.2f} min) - {n_unblur_events} unblur events",
            color=color_time)
        q_and_a_dict = get_questions_and_answers(task)
        prLog.print(f"Question: {q_and_a_dict['question']}")
        color_answer = "red" if q_and_a_dict["answer"].strip() == "" else "green"
        prLog.print(f"Answer: {q_and_a_dict['answer']}", color=color_answer)
        prLog.print(f"Raw answer: {q_and_a_dict['raw_answer']}")
    if len(all_task_times) > 0:
        avg_time_sec = sum(all_task_times) / len(all_task_times)
        avg_time_min = avg_time_sec / 60
        prLog.print(f"Average time: {avg_time_sec} sec ({avg_time_min:.2f} min)")
    if len(all_task_n_events) > 0:
        prLog.print(f"Average number of unblur events: {sum(all_task_n_events) / len(all_task_n_events)}")
    if output_filepath:
        prLog.dump()