import json
import sys
import time
from pymongo import MongoClient


def _replace_id_object(data):
    for row in data:
        row["_id"] = str(row["_id"])
    return data


def load_data_for(mongo_client, nickname):
    """Get all the records of a specific username."""
    edit_collection = mongo_client.participant_logs["edit_v01"]
    data = edit_collection.find({"nickname": nickname})
    data = list(data)
    data = _replace_id_object(data)
    return data


def sanitize(filename):
    """Sanitize the input string by removing backslashes and dots."""
    return filename.replace("\\", "").replace(".", "")

def load_all_aggregated_records(
        mongo_client,
        min_age_in_seconds: int = 0,
        keep_only_submitted: bool = True,
        remove_warmup: bool = True,
        ):
    """Get the aggregated records in MongoDB for each human.

    min_age_in_seconds: defines how old the returned record should be, e.g.
        if 3600 seconds it returns only records which are older than 1 hour
        excluding the most recent ones, since they might still be ongoing.
    """
    edit_collection = mongo_client.participant_logs["edit_v01"]
    # group by the nickname and sourceFile
    query = [
        {"$sort": { "time": 1}},
        {"$group": {
            "_id": {
                "nickname": "$nickname",
                "sourceFile": "$sourceFile",
                "platformname": "$platformname",
                "platformversion": "$platformversion",
                "completejsexecution": "$completejsexecution",
                "randomcode": "$randomcode",
                },
            "editableLineContent": {"$last": "$editableLineContent"},
            "version": {"$last": "$version"},
            "openedPageTime": {"$first": "$openedPageTime"},
            "formattedcode": {"$last": "$formattedcode"},
            "codeAreaIsVisible": {"$last": "$codeAreaIsVisible"},
            "platformos": {"$first": "$platformos"},
            # merge all event fields
            "options": {"$push": "$options"},
            "mouseHoverEvents": {"$push": "$mouseHoverEvents"},
            "hoverUnblurEvents": {"$push": "$hoverUnblurEvents"},
            "clickUnblurEvents": {"$push": "$clickUnblurEvents"},
            "editEvents": {"$push": "$editEvents"},
            "pasteEvents": {"$push": "$pasteEvents"},
            "searches": {"$push": "$searches"},
            "hintHovers": {"$push": "$hintHovers"},
            # create arrays for these fields
            "time": {"$push": "$time"},
            "flushReason": {"$push": "$flushReason"},
        }
    }]
    results = edit_collection.aggregate(query, allowDiskUse=True)
    results = list(results)
    # remove the groupby key fields from the _id and bring them to the
    # record itself
    results = [
        {
            **aggregated_record["_id"],
            **{
                k: v for k, v in aggregated_record.items()
                if k != "_id"
            }
        }
        for aggregated_record in results]
    #discard the records that are without nicknames
    results = [r for r in results if r["nickname"]]
    # remove the warmup
    if remove_warmup:
        results = [
            r for r in results if "instructions" not in r["sourceFile"]]
    # keep only submitted ones (aka completed)
    if keep_only_submitted:
        results = [
            r for r in results
            if "experiment-submit" in r["flushReason"]]
    if min_age_in_seconds > 0:
        _current_time = int(time.time())
        results = [
            r for r in results
            if _current_time - (int(r["openedPageTime"]) / 1000) > min_age_in_seconds]
    # add id field
    results = [
        {"local_filename":
            f"{sanitize(x['nickname'].replace('_', ''))}_" +
            f"{sanitize(x['sourceFile'].replace('_', ''))}_" +
            f"{sanitize(x['randomcode'])}",
            **x
        } for x in results
    ]
    # join all the lists of events
    for record in results:
        for event_filed in ["options", "mouseHoverEvents", "hoverUnblurEvents",
                            "clickUnblurEvents", "editEvents", "pasteEvents",
                            "searches", "hintHovers"]:
            # flatten the event_field
            record[event_filed] = [
                event for event_list in record[event_filed]
                for event in event_list]
    #for res in results:
        #print(res)
        #print("nickname: ", res["nickname"], " - sourceFile: ", res["sourceFile"])
    return results


def get_distinct_usernames(mongo_client):
    """Retrieve the unique username in the dataset."""
    edit_collection = mongo_client.participant_logs["edit_v01"]
    nicknames = edit_collection.find(
        {"flushReason": {"$ne": "check-flush"}}, {"nickname": 1})
    nicknames = [x["nickname"] for x in list(nicknames)]
    return set(nicknames)


def check_user_data(mongo_client):
    """Check if users completed the task or not."""
    report = {}
    names = list(get_distinct_usernames(mongo_client))
    for name in names:
        user_data = load_data_for(mongo_client, name)
        completed = [x for x in user_data if x["flushReason"] != "check-flush"]
        report[name] = [x["sourceFile"] for x in completed]
    return report


def delete_for_nickname(mongo_client, nickname):
    """Delete all the records of a specific username."""
    edit_collection = mongo_client.participant_logs["edit_v01"]
    edit_collection.delete_many({"nickname": nickname})


def pull_ratings(mongo_client):
    """Get the ratings for all the users."""
    rating_collection = mongo_client.participant_logs["rating_edit_v01"]
    records = rating_collection.find({})
    records = list(records)
    return _replace_id_object(records)


# Argument structure: <intention> <file output: True/False>
# Get all data for one user: nickname True/False <nickname>
# Get distinct names in edits collection: distinct-users True/False
# Get report of complete data per user: check-data True/False
# Delete edit data for nickname: delete <nickname>
if __name__ == "__main__":
    args = sys.argv[1:]
    intention = args[0]
    output_to_file = bool(args[1])
    fetched_data = None
    print_result = True
    out_name = "mongodb-data.json"

    if intention == "nickname":
        nickname = args[2]
        print(f"Fetching data for user {nickname}.")
        fetched_data = load_data_for(nickname)
    elif intention == "distinct-users":
        print("Getting distinct usernames.")
        fetched_data = get_distinct_usernames()
        fetched_data = list(fetched_data)
        out_name = "hrr-distinct-users.json"
    elif intention == "check-data":
        fetched_data = check_user_data()
    elif intention == "delete":
        nickname = args[1]
        delete_for_nickname(nickname)
        output_to_file = False
        fetched_data = f"Deleted data for {nickname}."
    elif intention == "final-results":
        print_result = False
        fetched_data = load_all_aggregated_records()
        out_name = args[2]
        print("Fetched final results.")
    elif intention == "pull-ratings":
        print_result = False
        fetched_data = pull_ratings()
        out_name = "hrr-ratings.json"
        print("Fetched rating data.")

    if print_result:
        print(fetched_data)

    if output_to_file:
        with open(out_name, "w") as f:
            json.dump(fetched_data, f, indent=4)