import os
import json
from nanoid import generate

def generate_folder_id():
    return generate(size=21)

def create_folder_tree(path, parent_id=None, parent_id_list=[]):
    folder_name = os.path.abspath(path)  # 절대 경로를 폴더 이름으로 설정
    folder_id = generate_folder_id()
    
    # 현재 폴더 정보를 설정
    folder_info = {
        "id": folder_id,
        "name": folder_name,
        "children": [],
        "parentId": parent_id,
        "parentIdList": parent_id_list
    }

    # 부모 ID 리스트를 갱신
    parent_id_list_updated = parent_id_list + [folder_id] if parent_id else [folder_id]

    # 폴더 딕셔너리 생성
    folder_dict = {folder_id: folder_info}
    
    # 하위 폴더들을 처리
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            child_folder_result = create_folder_tree(item_path, folder_id, parent_id_list_updated)
            child_folder_info = child_folder_result["folder_info"]
            folder_info["children"].append(child_folder_info["id"])
            folder_dict.update(child_folder_result["folder_dict"])

    return {
        "folder_info": folder_info,
        "folder_dict": folder_dict
    }

def generate_folder_structure(root_path):
    result = create_folder_tree(root_path)
    folder_dict = result["folder_dict"]
    return folder_dict

# Example usage
root_path = '/Users/nerdystar/Desktop/test_folder_tree'
folder_structure = generate_folder_structure(root_path)

# Save to JSON file
with open('folder_structure.json', 'w', encoding='utf-8') as f:
    json.dump(folder_structure, f, ensure_ascii=False, indent=4)
