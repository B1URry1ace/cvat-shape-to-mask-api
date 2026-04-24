import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# Новые библиотеки для скачивания и распаковки
import zipfile
import os
import tempfile

from cvat_sdk import make_client
from cvat_sdk.models import PatchedLabeledDataRequest, LabeledShapeRequest

# ===================== НАСТРОЙКИ =====================
CVAT_URL = "http://localhost:8080"
TASK_ID = #<----------------------------------------------------------------------------   Номер вашего таска

USERNAME = #<----------------------------------------------------------------------------  ЗАПОЛНИТЬ
PASSWORD = #<----------------------------------------------------------------------------  ЗАПОЛНИТЬ
# ====================================================

def rle_encode(mask):
    """Идеально чистый конвертер в RLE-маску"""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None
        
    xtl, ytl = int(x_indices.min()), int(y_indices.min())
    xbr, ybr = int(x_indices.max()), int(y_indices.max())
    
    if (xbr - xtl) < 3 or (ybr - ytl) < 3:
        return None
        
    cropped_mask = mask[ytl:ybr+1, xtl:xbr+1]
    flat = cropped_mask.flatten()
    
    diffs = np.where(flat[1:] != flat[:-1])[0]
    run_lengths = np.diff(np.concatenate(([-1], diffs, [len(flat) - 1])))
    
    runs =[float(r) for r in run_lengths] 
    if flat[0] == 1:
        runs.insert(0, 0.0)
        
    return runs +[float(xtl), float(ytl), float(xbr), float(ybr)]

def upload_masks_directly():
    print("1. Подключаемся к CVAT по API...")
    with make_client(host=CVAT_URL, credentials=(USERNAME, PASSWORD)) as client:
        task = client.tasks.retrieve(TASK_ID)
        
        # --- НОВЫЙ БЛОК: СКАЧИВАЕМ XML АВТОМАТИЧЕСКИ ---
        print("\n2. Скачиваем актуальный annotations.xml (полигоны) с сервера...")
        
        # Создаем временную папку, которая сама удалится после выхода из блока
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "annotations.zip")
            
            # Заставляем CVAT сгенерировать и отдать нам датасет (только разметку)
            task.export_dataset(
                format_name="CVAT for images 1.1",
                filename=zip_path,
                include_images=False
            )
            
            # Распаковываем скачанный ZIP архив
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("annotations.xml", path=tmpdirname)
                
            # Читаем XML прямо в память
            xml_path = os.path.join(tmpdirname, "annotations.xml")
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
        print("  ✓ Файл успешно скачан, загружен в память и удалён с диска!")
        # -----------------------------------------------
        
        print("\n3. Получаем текущие аннотации (для защиты от дубликатов)...")
        current_anns = client.tasks.api.retrieve_annotations(id=TASK_ID)
        if isinstance(current_anns, tuple): 
            current_anns = current_anns[0]
            
        existing_shapes = current_anns.shapes if hasattr(current_anns, 'shapes') else current_anns.get('shapes',[])
        
        existing_signatures = set()
        for shape in existing_shapes:
            s_type = getattr(shape, 'type', shape.get('type') if isinstance(shape, dict) else None)
            s_type_val = s_type.value if hasattr(s_type, 'value') else s_type
            
            if s_type_val == "mask":
                s_frame = getattr(shape, 'frame', shape.get('frame') if isinstance(shape, dict) else None)
                s_label_id = getattr(shape, 'label_id', shape.get('label_id') if isinstance(shape, dict) else None)
                s_points = getattr(shape, 'points', shape.get('points') if isinstance(shape, dict) else None)
                
                if s_points is not None:
                    pts_tuple = tuple(int(p) for p in s_points)
                    sig = (s_frame, s_label_id, pts_tuple)
                    existing_signatures.add(sig)
                    
        print(f"  [ИНФО] В задаче уже есть фигур: {len(existing_shapes)}.")

        labels_in_task = task.get_labels()
        if not labels_in_task:
            print("\n[ОШИБКА] В Задаче CVAT нет ни одного класса!")
            return
            
        label_map = {lbl.name: lbl.id for lbl in labels_in_task}
        fallback_label_id = labels_in_task[0].id 
        
        meta = client.tasks.api.retrieve_data_meta(id=TASK_ID)
        if isinstance(meta, tuple): meta = meta[0]
        
        name2frame = {}
        frames_list = meta.frames if hasattr(meta, 'frames') else meta['frames']
        for idx, f in enumerate(frames_list):
            filename = getattr(f, 'name', f.get('name') if isinstance(f, dict) else None)
            clean_name = Path(filename).stem
            name2frame[clean_name] = {
                "idx": idx,
                "width": getattr(f, 'width', f.get('width') if isinstance(f, dict) else None),
                "height": getattr(f, 'height', f.get('height') if isinstance(f, dict) else None)
            }
            
        print("\n4. Конвертируем скачанные полигоны в RLE-маски...")
        new_shapes =[]
        skipped_frames = 0
        processed_polygons = 0
        skipped_duplicates = 0

        for image in root.findall("image"):
            clean_name = Path(image.get("name")).stem
            
            if clean_name not in name2frame:
                skipped_frames += 1
                continue
                
            frame_info = name2frame[clean_name]
            frame_idx = frame_info["idx"]
            real_width = frame_info["width"]
            real_height = frame_info["height"]
            
            for poly in image.findall("polygon"):
                label_name = poly.get("label")
                if label_name == "background":
                    continue
                    
                target_label_id = label_map.get(label_name, fallback_label_id)
                
                pts_str = poly.get("points")
                pts =[]
                for p in pts_str.split(";"):
                    x, y = map(float, p.split(","))
                    x = max(0, min(int(x), real_width - 1))
                    y = max(0, min(int(y), real_height - 1))
                    pts.append([x, y])
                    
                pts_arr = np.array(pts, dtype=np.int32)
                
                mask = np.zeros((real_height, real_width), dtype=np.uint8)
                cv2.fillPoly(mask, [pts_arr], color=1)
                
                rle_points = rle_encode(mask)
                if rle_points is None:
                    continue
                    
                pts_tuple = tuple(int(p) for p in rle_points)
                sig = (frame_idx, target_label_id, pts_tuple)
                
                if sig in existing_signatures:
                    skipped_duplicates += 1
                    continue
                    
                existing_signatures.add(sig) 
                
                new_shapes.append(
                    LabeledShapeRequest(
                        type="mask", 
                        frame=frame_idx,
                        label_id=target_label_id,
                        points=rle_points,
                        z_order=int(poly.get("z_order", 0))
                    )
                )
                processed_polygons += 1

        print(f"  ✓ Сгенерировано новых масок: {processed_polygons}")
        if skipped_duplicates > 0:
            print(f"  ✓ Пропущено дубликатов (уже были в CVAT): {skipped_duplicates}")
        
        if len(new_shapes) == 0:
            print("\n[ИНФО] Все полигоны уже сконвертированы. Новых масок нет.")
            return

        print("\n5. Обновляем базу CVAT (отправляем только новые маски)...")
        client.tasks.api.partial_update_annotations(
            id=TASK_ID,
            action="create",
            patched_labeled_data_request=PatchedLabeledDataRequest(
                shapes=new_shapes, 
                tracks=[], 
                tags=[], 
            )
        )

    print("\nГОТОВО! Новые маски успешно добавлены к существующим аннотациям.")

if __name__ == "__main__":
    upload_masks_directly()