from Design_Description_Text_Extraction import split_dxf_to_png, image_to_txt, merge_txt_in_folder
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

dxf_path = os.path.join(SCRIPT_DIR, "Design_Description_Text_Extraction", "interest.dxf")
output_dir = os.path.join(SCRIPT_DIR, "Design_Description_Text_Extraction", "fontstest-2")
# split_dxf_to_png(dxf_path, output_dir)



txt_root = os.path.join(output_dir, "txt")
os.makedirs(txt_root, exist_ok=True)

frames = [
    name for name in os.listdir(output_dir)
    if name.startswith("frame_") and os.path.isdir(os.path.join(output_dir, name))
]

# with ThreadPoolExecutor(max_workers=8) as executor:
#     future_to_frame = {}
#     for frame_name in frames:
#         frame_dir = os.path.join(output_dir, frame_name)
#         output_txt = os.path.join(txt_root, f"{frame_name}_recognition_result.txt")
#         future = executor.submit(image_to_txt, frame_dir, output_txt)
#         future_to_frame[future] = frame_name

#     for future, frame_name in future_to_frame.items():
#         try:
#             future.result()
#             print(f"{frame_name} done")
#         except Exception as e:
#             print(f"{frame_name} error: {e}")


output_txt = os.path.join(txt_root, "merged_all_frames.txt")
merge_txt_in_folder(txt_root, output_txt)
