from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_OBJECTS = ("car", "motorcycle", "bus", "truck")
WINDOW_NAME = "YOLO object counter"


@dataclass(frozen=True)
class CountEvent:
    frame: int
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    center_x: int
    center_y: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count vehicles or other COCO objects in a video with Ultralytics YOLO26 "
            "detection/segmentation models and object tracking."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video, webcam index such as 0, or a stream URL.",
    )
    parser.add_argument(
        "--output",
        default="runs/counted_video.mp4",
        help="Path for annotated output video. Use an empty string to disable saving.",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO model weights, for example yolo26n.pt or yolo26n-seg.pt.",
    )
    parser.add_argument(
        "--objects",
        default=",".join(DEFAULT_OBJECTS),
        help=(
            "Comma-separated object names or COCO class ids to count. "
            "Default: car,motorcycle,bus,truck."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("unique", "line"),
        default="line",
        help="unique counts each track once; line counts each track once after crossing a line.",
    )
    parser.add_argument(
        "--line",
        default="",
        help=(
            "Counting line as x1,y1,x2,y2. If omitted in line mode, a horizontal "
            "line is placed at 55%% of frame height."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold used by YOLO.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--device",
        default="",
        help="Device for inference: empty for auto, cpu, 0, 0,1, etc.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config, for example bytetrack.yaml or botsort.yaml.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated frames while processing.",
    )
    parser.add_argument(
        "--csv",
        default="runs/count_events.csv",
        help="Path to save counted events CSV. Use an empty string to disable.",
    )
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def normalize_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: name for idx, name in enumerate(names)}


def resolve_class_ids(raw_objects: str, names: dict[int, str]) -> list[int]:
    name_to_id = {name.lower(): class_id for class_id, name in names.items()}
    result: list[int] = []

    for token in (item.strip() for item in raw_objects.split(",")):
        if not token:
            continue
        if token.isdigit():
            class_id = int(token)
            if class_id not in names:
                raise ValueError(f"Class id {class_id} is absent in model names.")
            result.append(class_id)
            continue

        key = token.lower()
        if key not in name_to_id:
            available = ", ".join(names[i] for i in sorted(names)[:15])
            raise ValueError(
                f"Unknown class name '{token}'. First available classes: {available}."
            )
        result.append(name_to_id[key])

    if not result:
        raise ValueError("At least one object class must be selected.")

    return sorted(set(result))


def parse_line(raw_line: str, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if not raw_line:
        y = int(height * 0.55)
        return (0, y), (width - 1, y)

    parts = [int(value.strip()) for value in raw_line.split(",")]
    if len(parts) != 4:
        raise ValueError("--line must contain four integers: x1,y1,x2,y2.")

    x1, y1, x2, y2 = parts
    return (x1, y1), (x2, y2)


def point_side(point: tuple[int, int], line: tuple[tuple[int, int], tuple[int, int]]) -> int:
    (x1, y1), (x2, y2) = line
    px, py = point
    value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    color = rng.integers(80, 240, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> None:
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 0.5
    overlay = np.zeros_like(frame)
    overlay[mask_bool] = color
    frame[mask_bool] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)[mask_bool]


def draw_hud(
    frame: np.ndarray,
    mode: str,
    total: int,
    class_counts: Counter[str],
    line: tuple[tuple[int, int], tuple[int, int]] | None,
) -> None:
    if line is not None:
        cv2.line(frame, line[0], line[1], (0, 255, 255), 3)

    rows = [f"Mode: {mode}", f"Total: {total}"]
    rows.extend(f"{name}: {count}" for name, count in sorted(class_counts.items()))
    panel_height = 26 + len(rows) * 26
    cv2.rectangle(frame, (12, 12), (300, panel_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (12, 12), (300, panel_height), (0, 255, 255), 1)

    for idx, text in enumerate(rows):
        cv2.putText(
            frame,
            text,
            (24, 42 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def write_events_csv(path: str, events: Iterable[CountEvent]) -> None:
    if not path:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "center_x",
                "center_y",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "frame": event.frame,
                    "track_id": event.track_id,
                    "class_id": event.class_id,
                    "class_name": event.class_name,
                    "confidence": f"{event.confidence:.4f}",
                    "center_x": event.center_x,
                    "center_y": event.center_y,
                }
            )


def process_video(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    names = normalize_names(model.names)
    class_ids = resolve_class_ids(args.objects, names)

    capture = open_source(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    ok, first_frame = capture.read()
    if not ok:
        raise RuntimeError("Video source opened, but the first frame could not be read.")

    height, width = first_frame.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    line = parse_line(args.line, width, height) if args.mode == "line" else None

    writer: cv2.VideoWriter | None = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    seen_ids: set[int] = set()
    crossed_ids: set[int] = set()
    previous_side: dict[int, int] = {}
    class_counts: Counter[str] = Counter()
    events: list[CountEvent] = []
    frame_index = 0
    pending_frame: np.ndarray | None = first_frame

    while True:
        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
        else:
            ok, frame = capture.read()
            if not ok:
                break

        frame_index += 1
        result = model.track(
            frame,
            persist=True,
            classes=class_ids,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            tracker=args.tracker,
            device=args.device or None,
            verbose=False,
        )[0]

        annotated = frame.copy()
        boxes = result.boxes
        masks = None
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)
            class_values = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for det_idx, (box, track_id, class_id, confidence) in enumerate(
                zip(xyxy, track_ids, class_values, confidences)
            ):
                x1, y1, x2, y2 = box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                class_name = names.get(class_id, str(class_id))
                color = color_for_id(track_id)

                if masks is not None and det_idx < len(masks):
                    draw_mask(annotated, masks[det_idx], color)

                counted = False
                if args.mode == "unique":
                    if track_id not in seen_ids:
                        seen_ids.add(track_id)
                        counted = True
                elif line is not None:
                    side = point_side(center, line)
                    old_side = previous_side.get(track_id)
                    if old_side is None:
                        previous_side[track_id] = side
                    elif side != 0 and old_side != 0 and side != old_side:
                        previous_side[track_id] = side
                        if track_id not in crossed_ids:
                            crossed_ids.add(track_id)
                            counted = True
                    elif side != 0:
                        previous_side[track_id] = side

                if counted:
                    class_counts[class_name] += 1
                    events.append(
                        CountEvent(
                            frame=frame_index,
                            track_id=track_id,
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(confidence),
                            center_x=center[0],
                            center_y=center[1],
                        )
                    )

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, center, 4, color, -1)
                label = f"#{track_id} {class_name} {confidence:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        total = len(seen_ids) if args.mode == "unique" else len(crossed_ids)
        draw_hud(annotated, args.mode, total, class_counts, line)

        if writer is not None:
            writer.write(annotated)
        if args.show:
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    write_events_csv(args.csv, events)

    print("Processing finished.")
    print(f"Mode: {args.mode}")
    print(f"Total counted objects: {len(seen_ids) if args.mode == 'unique' else len(crossed_ids)}")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")
    if args.output:
        print(f"Annotated video: {Path(args.output).resolve()}")
    if args.csv:
        print(f"CSV events: {Path(args.csv).resolve()}")


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
