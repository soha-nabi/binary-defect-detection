import os
import shutil
import xml.etree.ElementTree as ET

SOURCE = "dataset/NEU-DET"
TARGET = "dataset_binary"

TARGET_DEFECT = "scratches"  # ⭐ choose this as defect class


for split in ["train", "validation"]:
    img_dir = os.path.join(SOURCE, split, "images")
    ann_dir = os.path.join(SOURCE, split, "annotations")

    for img_file in os.listdir(img_dir):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(img_dir, img_file)

        xml_file = img_file.replace(".jpg", ".xml")
        xml_path = os.path.join(ann_dir, xml_file)

        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        label = "non_defect"

        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            if TARGET_DEFECT in name:
                label = "defect"
                break

        dest_dir = os.path.join(TARGET, split, label)
        os.makedirs(dest_dir, exist_ok=True)

        shutil.copy(img_path, dest_dir)

print("Dataset conversion completed ✅")