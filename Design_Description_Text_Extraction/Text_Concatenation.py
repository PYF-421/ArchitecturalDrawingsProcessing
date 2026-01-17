import os


def merge_txt_in_folder(input_dir: str, output_path: str) -> None:
    files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".txt")
    ]
    files.sort()
    with open(output_path, "w", encoding="utf-8") as out_f:
        for name in files:
            path = os.path.join(input_dir, name)
            with open(path, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())
                out_f.write("\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "fontstest-2", "txt")
    output_txt = os.path.join(input_dir, "merged_all_frames.txt")
    merge_txt_in_folder(input_dir, output_txt)

