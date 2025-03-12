import os
import glob
import json
import argparse


def get_json_filenames(summaries_dir: str, lang: str) -> list:
    glob_str = os.path.join(summaries_dir, lang, "*.json")
    json_filenames = glob.glob(glob_str)
    json_filenames = [f for f in json_filenames if "_completed.json" not in f]
    return json_filenames

def read_summary_as_dict(json_filename: str) -> dict:
    with open(json_filename, "r", encoding="utf-8") as file:
        summary_dict = json.load(file)
    return summary_dict

def save_summary_dict(summary_dict: dict, json_filename: str) -> None:
    """
    Saves a dictionary to a JSON file with the specified filename.
    
    Args:
        summary_dict (dict): Dictionary to save.
        json_filename (str): The file path where the JSON should be saved.
    """
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(summary_dict, json_file, ensure_ascii=False, indent=4)  # Pretty formatting

def extract_text_after_newline(summary_dict: dict) -> str:
    summary_text = summary_dict.get("summary", "")
    if "\n" in summary_text:
        return summary_text.split("\n", 1)[1]  # Split at first newline and return the second part
    return ""  # Return empty string if no newline found

def print_before_after_newline(summary_dict: dict) -> None:
    summary_split_by_new_line = summary_dict["summary"].split("\n", 1)
    if len(summary_split_by_new_line) == 1:
        print("before newline: \n", summary_split_by_new_line[0])
        print("after newline: \n", "None")
    elif len(summary_split_by_new_line) == 2:
        print("before newline: \n", summary_split_by_new_line[0])
        print("after newline: \n", summary_split_by_new_line[1])
    else:
        raise ValueError
    
def remove_summary_after_newline(summary_dict: dict) -> dict:
    """
    Removes everything after the first newline (\n) in the 'summary' key of the input dictionary.
    
    Args:
        summary_dict (dict): Input dictionary containing a 'summary' key.
    
    Returns:
        dict: A new dictionary with the modified 'summary' value.
    """
    modified_dict = summary_dict.copy()  # Create a copy to avoid modifying the original dict
    if "summary" in modified_dict and "\n" in modified_dict["summary"]:
        modified_dict["summary"] = modified_dict["summary"].split("\n", 1)[0]  # Keep only text before \n
    return modified_dict

def get_json_filename_to_save(json_filename: str, save_dir: str, lang: str) -> str:
    """
    Extracts the basename from json_filename and joins it with save_dir.
    Ensures that save_dir exists before returning the full path.
    
    Args:
        json_filename (str): The original file path.
        save_dir (str): The directory where the new file should be saved.
        lang : (str)

    Returns:
        str: The new file path in save_dir.
    """
    # Ensure save_dir exists
    save_dir_lang = os.path.join(save_dir, lang)
    os.makedirs(save_dir_lang, exist_ok=True)

    # Extract the basename (e.g., '7f9eed2161.json') and join with save_dir
    json_filename_to_save = os.path.join(save_dir_lang, os.path.basename(json_filename))
    return json_filename_to_save

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate summaries for multilingual Wikipedia articles")
    
    parser.add_argument("--summaries-dir", required=True,
                        help="Input directory of summaries")
    parser.add_argument("--languages", nargs="+", default=["en", "fr", "de", "ja"],
                        help="Languages to process (default: en fr de ja ru)")
    parser.add_argument("--save-data-dir", required=False,
                        help="Directory to save summary json. If it's not specified, the original json will be overwritten.")
    args = parser.parse_args()
    
    summaries_dir = args.summaries_dir
    if args.save_data_dir is None:
        save_dir = summaries_dir
    else:
        save_dir = args.save_data_dir

    for lang in args.languages:
        print("lang : ", lang)
        json_filenames = get_json_filenames(summaries_dir, lang)

        # for json_filename in json_filenames:
        #     print(json_filename)
        #     summary_dict = read_summary_as_dict(json_filename)
        #     print_before_after_newline(summary_dict)

        for json_filename in json_filenames:
            print("processing ", json_filename)
            summary_dict = read_summary_as_dict(json_filename)
            _summary_dict = remove_summary_after_newline(summary_dict)
            json_filename_to_save = get_json_filename_to_save(json_filename, save_dir, lang)
            save_summary_dict(_summary_dict, json_filename_to_save)


if __name__ == "__main__":
    main()