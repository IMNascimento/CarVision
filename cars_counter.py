import argparse
import time
from typing import Dict, Tuple
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

# ----------------------------
# Configs do pipeline
# ----------------------------
CLASSES_COCO_INTERESSE = {"car", "truck", "bus", "motorcycle"}  
YOLO_WEIGHTS = "yolov8n.pt" 
CONF_THRES = 0.40
IOU_THRES = 0.5
IMGSZ = 640

# Anti-ruído para contagem por ID
MIN_FRAMES_BEFORE_COUNT = 3   # qtos frames o ID precisa existir antes de contar
MIN_CONF_TO_COUNT = 0.35      # confiança mínima no frame do “primeiro count”

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="0 (webcam), caminho de vídeo ou RTSP (ex.: rtsp://user:pass@ip:554/...)")
    p.add_argument("--save", action="store_true", help="Salvar vídeo anotado")
    p.add_argument("--out", default="saida_annotated.mp4", help="Arquivo de saída (se --save)")
    p.add_argument("--show", action="store_true", help="Mostrar janela ao vivo")
    p.add_argument("--hide-labels", action="store_true", help="Não desenhar labels")
    p.add_argument("--hide-trace", action="store_true", help="Não desenhar trilhas")
    return p.parse_args()

def main():
    args = parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    model = YOLO(YOLO_WEIGHTS)
    names = model.model.names
    classe_ids_interesse = [i for i, n in names.items() if n in CLASSES_COCO_INTERESSE]

    tracker = ByteTrack()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=20)


    class_totals = defaultdict(int)   
    total_all = 0

    # Estados por track_id
    track_age: Dict[int, int] = defaultdict(int)        
    counted_ids: Dict[int, bool] = defaultdict(bool)    
    last_class_by_id: Dict[int, str] = {}               
    last_conf_by_id: Dict[int, float] = {}              

    last_time = time.time()
    frames = 0
    fps_now = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMGSZ,
            verbose=False
        )

        dets_list = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            if len(class_ids) > 0:
                mask = np.isin(class_ids, classe_ids_interesse)
                boxes_xyxy = boxes_xyxy[mask]
                confidences = confidences[mask]
                class_ids = class_ids[mask]

            det = sv.Detections(
                xyxy=boxes_xyxy,
                confidence=confidences,
                class_id=class_ids
            )
            dets_list.append(det)

        detections = sv.Detections.empty() if len(dets_list) == 0 else (
            dets_list[0] if len(dets_list) == 1 else sv.Detections.merge(dets_list)
        )

        tracked = tracker.update_with_detections(detections)

        anchors = tracked.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        tids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)
        cids = tracked.class_id if tracked.class_id is not None else np.array([], dtype=int)
        confs = tracked.confidence if tracked.confidence is not None else np.array([], dtype=float)

        for i in range(len(tracked)):
            if i >= len(tids):
                continue
            tid = int(tids[i])
            track_age[tid] += 1

            if i < len(cids):
                cls_idx = int(cids[i])
                cls_name = names.get(cls_idx, "obj")
                last_class_by_id[tid] = cls_name
            if i < len(confs):
                last_conf_by_id[tid] = float(confs[i])

            if not counted_ids[tid] and track_age[tid] >= MIN_FRAMES_BEFORE_COUNT:
                cls_name = last_class_by_id.get(tid, None)
                conf_val = last_conf_by_id.get(tid, 0.0)
                if cls_name in CLASSES_COCO_INTERESSE and conf_val >= MIN_CONF_TO_COUNT:
                    class_totals[cls_name] += 1
                    total_all += 1
                    counted_ids[tid] = True  


        labels = []
        for i in range(len(tracked)):
            cls_name = names[int(cids[i])] if i < len(cids) else "obj"
            conf = float(confs[i]) if i < len(confs) else 0.0
            tid = int(tids[i]) if i < len(tids) else -1
            labels.append(f"#{tid} {cls_name} {conf:.2f}")

        # Desenho
        frame = box_annotator.annotate(scene=frame, detections=tracked)
        if not args.hide_labels:
            frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)
        if not args.hide_trace:
            frame = trace_annotator.annotate(scene=frame, detections=tracked)


        x0 = 10
        y0 = 10
        w = 370
        h = 160
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
        cv2.putText(frame, f"Total: {total_all}", (x0 + 10, y0 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        row = 0
        for cls in ["car", "truck", "bus", "motorcycle"]:
            cv2.putText(frame, f"{cls}: {class_totals.get(cls, 0)}",
                        (x0 + 10, y0 + 70 + row * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            row += 1

        # FPS
        frames += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps_now = frames / (now - last_time)
            last_time = now
            frames = 0
        cv2.putText(frame, f"FPS: {fps_now:.1f}", (width - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if args.show:
            cv2.imshow("Contador (Total por Classe)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n==== RESUMO ====")
    print(f"Total geral: {total_all}")
    for cls in ["car", "truck", "bus", "motorcycle"]:
        print(f"{cls:12s}: {class_totals.get(cls, 0)}")

if __name__ == "__main__":
    main()
