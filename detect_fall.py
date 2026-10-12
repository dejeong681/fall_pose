from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from pose_fall import draw_pose, inspect_fall


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTION_MODEL = SCRIPT_DIR / "fall_20261007.pt"
POSE_MODEL = SCRIPT_DIR / "yolo11n-pose.pt"
DEVICE = "cpu"
DET_CONF = 0.25
POSE_CONF = 0.25
# 임시 실험값입니다. crop 여유와 키포인트 기준은 실제 데이터로 조정합니다.
CROP_MARGIN = 0.25
KP_CONF = 0.5


def apply_fall_action(box, action, person_id, person_confidence=None):
    """최종 박스를 정합니다. 추후 정상 자세 판정에서도 이 함수를 사용합니다."""
    if action == "remove":
        return None
    updated = box.clone()
    if action == "person":
        if person_id is None or person_confidence is None:
            raise ValueError("person 변경에는 person 클래스와 포즈 사람 신뢰도가 필요합니다.")
        updated[-1] = person_id
        updated[-2] = person_confidence
    elif action != "keep":
        raise ValueError(f"알 수 없는 박스 처리: {action}")
    return updated


def save_image(path, image):
    """한글 경로에서도 결과 이미지를 저장합니다."""
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    encoded.tofile(path)


def run(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"폴더를 찾을 수 없습니다: {folder}")

    # 두 모델을 한 번만 불러옵니다. fall 후보가 없으면 포즈를 실행하지 않습니다.
    model = YOLO(str(DETECTION_MODEL))
    pose_model = YOLO(str(POSE_MODEL))
    if pose_model.task != "pose":
        raise ValueError("POSE_MODEL에는 포즈 모델을 지정하세요.")

    # 모델의 클래스 이름으로 fall ID를 찾습니다.
    fall_ids = {
        class_id for class_id, name in model.names.items()
        if name.strip().lower() == "fall"
    }
    if not fall_ids:
        raise ValueError("모델에 fall 클래스가 없습니다.")
    person_id = next((index for index, name in model.names.items() if name.lower() == "person"), None)

    # 기존 결과를 덮어쓰지 않고 실행마다 새 폴더를 사용합니다.
    output_base = SCRIPT_DIR / "results" / f"{folder.name}_fall_pose"
    output_dir = output_base
    suffix = 2
    while output_dir.exists():
        output_dir = output_base.with_name(f"{output_base.name}_{suffix}")
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)

    image_count = 0
    fall_image_count = 0
    total_fall_count = 0
    remaining_image_count = 0
    remaining_fall_count = 0
    removed_fall_count = 0

    for result in model.predict(
        source=str(folder),
        conf=DET_CONF,
        device=DEVICE,
        save=False,
        stream=True,
        verbose=False,
    ):
        image_count += 1
        final_name = f"{image_count:04d}_{Path(result.path).stem}.jpg"
        boxes = result.boxes.data if result.boxes is not None else None
        fall_count = sum(int(box[-1]) in fall_ids for box in boxes) if boxes is not None else 0
        if fall_count == 0:
            # fall이 없거나 객체가 하나도 없어도 전체 이미지 결과를 저장합니다.
            save_image(output_dir / final_name, result.plot())
            continue
        fall_image_count += 1
        total_fall_count += fall_count
        (output_dir / "crops").mkdir(parents=True, exist_ok=True)
        original = result.orig_img
        final_boxes = []
        retained_poses = []
        candidate_number = 0

        for box in boxes:
            if int(box[-1]) not in fall_ids:
                final_boxes.append(box.clone())
                continue
            candidate_number += 1
            # 박스가 그려지기 전의 원본에서 후보를 crop합니다.
            analysis = inspect_fall(
                original, box[:4].cpu().tolist(), pose_model,
                CROP_MARGIN, KP_CONF, POSE_CONF, DEVICE,
            )
            # 현재 규칙: 사용 가능한 점 0개면 제거, 1개 이상이면 fall 유지.
            # 추후 정상 자세 규칙에서 action="person" 또는 "remove"를 반환할 수 있습니다.
            final_box = apply_fall_action(
                box, analysis["action"], person_id, analysis["person_confidence"],
            )
            if final_box is None:
                removed_fall_count += 1
            else:
                final_boxes.append(final_box)
                retained_poses.append(analysis)

            crop_image = draw_pose(analysis["crop"].copy(), analysis["keypoints"], analysis["usable"])
            caption = f"{analysis['action'].upper()} | kp={analysis['keypoint_count']}/17"
            cv2.putText(crop_image, caption, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            crop_name = f"{image_count:04d}_{Path(result.path).stem}_fall{candidate_number}.jpg"
            save_image(output_dir / "crops" / crop_name, crop_image)
            print(
                f"[{fall_image_count}] {Path(result.path).name} | 후보 {candidate_number} "
                f"| 키포인트 {analysis['keypoint_count']}개 | {analysis['action']} "
                f"| {analysis['reason']}",
                flush=True,
            )

        # 기존 그림에서 지우지 않고, 최종 박스로 원본에 다시 그립니다.
        final_data = torch.stack(final_boxes) if final_boxes else boxes[:0].clone()
        result.update(boxes=final_data)
        current_falls = sum(int(box[-1]) in fall_ids for box in final_data)
        remaining_fall_count += current_falls
        remaining_image_count += int(current_falls > 0)
        final_image = result.plot()
        for analysis in retained_poses:
            draw_pose(final_image, analysis["keypoints"], analysis["usable"], analysis["crop_box"][:2])
        # fall 제거 여부와 관계없이 이 이미지의 최종 검출 결과를 저장합니다.
        save_image(output_dir / final_name, final_image)

    print(f"\n전체 처리 이미지: {image_count}장")
    print(f"fall이 검출된 이미지: {fall_image_count}장")
    print(f"포즈 검사 전 fall: {total_fall_count}개")
    print(f"키포인트 없음으로 제거: {removed_fall_count}개")
    print(f"포즈 검사 후 fall: {remaining_fall_count}개 / {remaining_image_count}장")
    if image_count:
        print(f"결과 폴더: {output_dir}")


if __name__ == "__main__":
    folder_path = input("원본 이미지 폴더 경로: ").strip().strip('"')
    if not folder_path:
        raise ValueError("폴더 경로를 입력하세요.")
    run(folder_path)
