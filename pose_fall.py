"""fall 후보의 포즈 처리. 실행은 detect_fall.py에서 합니다."""

import cv2
import numpy as np


# COCO 17 키포인트의 연결 관계입니다.
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def inspect_fall(image, fall_box, pose_model, margin, kp_conf, pose_conf, device):
    """후보 하나의 원본 crop과 포즈, 최종 박스 처리 방법을 반환합니다."""
    height, width = image.shape[:2]
    left, top, right, bottom = map(float, fall_box)
    box_width, box_height = right - left, bottom - top
    # 경계에서 잘린 실제 crop 좌표를 기록합니다.
    x1 = max(0, int(np.floor(left - box_width * margin)))
    y1 = max(0, int(np.floor(top - box_height * margin)))
    x2 = min(width, int(np.ceil(right + box_width * margin)))
    y2 = min(height, int(np.ceil(bottom + box_height * margin)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("fall 박스로 유효한 crop을 만들 수 없습니다.")
    crop = image[y1:y2, x1:x2].copy()
    analysis = {
        "crop": crop, "crop_box": (x1, y1, x2, y2),
        "keypoints": None, "usable": None, "keypoint_count": 0, "person_confidence": None,
        "action": "remove", "reason": "포즈 사람 미검출",
    }
    pose = pose_model.predict(
        source=crop, conf=pose_conf, device=device, save=False, verbose=False,
    )[0]
    if pose.boxes is None or len(pose.boxes) == 0:
        return analysis
    if pose.keypoints is None or len(pose.keypoints) == 0:
        analysis["reason"] = "키포인트 출력 없음"
        return analysis

    # 원래 fall 박스를 crop 좌표로 옮겨 포즈 사람 박스와 IoU를 비교합니다.
    # IoU가 같은 경우에는 fall 박스 중심에 가까운 사람을 선택합니다.
    candidate = np.array([left - x1, top - y1, right - x1, bottom - y1])
    boxes = pose.boxes.xyxy.cpu().numpy()
    intersection_wh = np.maximum(
        0, np.minimum(boxes[:, 2:], candidate[2:]) - np.maximum(boxes[:, :2], candidate[:2]),
    )
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    union = areas + candidate_area - intersection
    ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    center = (candidate[:2] + candidate[2:]) / 2
    distances = np.linalg.norm((boxes[:, :2] + boxes[:, 2:]) / 2 - center, axis=1)
    selected = max(range(len(boxes)), key=lambda index: (ious[index], -distances[index]))
    analysis["person_confidence"] = float(pose.boxes.conf[selected])
    points = pose.keypoints.data[selected].cpu().numpy().copy()
    if points.shape != (17, 3):
        raise ValueError("신뢰도를 포함한 COCO 17 키포인트 모델이 필요합니다.")

    crop_height, crop_width = crop.shape[:2]
    usable = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= kp_conf)
        & (points[:, 0] >= 0) & (points[:, 0] < crop_width)
        & (points[:, 1] >= 0) & (points[:, 1] < crop_height)
        & np.any(points[:, :2] != 0, axis=1)
    )
    count = int(usable.sum())
    analysis.update(
        keypoints=points, usable=usable, keypoint_count=count,
        action="keep" if count > 0 else "remove",
        reason="키포인트 있음" if count > 0 else "기준을 만족하는 키포인트 없음",
    )
    # 키포인트 존재 여부만 검사합니다. 정상 자세 판정은 아직 하지 않습니다.
    return analysis


def draw_pose(image, points, usable, offset=(0, 0)):
    """선택한 사람의 사용 가능한 키포인트를 crop 또는 원본에 표시합니다."""
    if points is None or usable is None:
        return image
    positions = points[:, :2] + np.array(offset)
    for first, second in SKELETON:
        if usable[first] and usable[second]:
            a = tuple(np.rint(positions[first]).astype(int))
            b = tuple(np.rint(positions[second]).astype(int))
            cv2.line(image, a, b, (255, 200, 0), 2)
    for index in np.flatnonzero(usable):
        point = tuple(np.rint(positions[index]).astype(int))
        cv2.circle(image, point, 3, (0, 255, 0), -1)
    return image
