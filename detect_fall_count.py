from pathlib import Path

from ultralytics import YOLO


if __name__ == "__main__":
    folder = Path(input("이미지 폴더 경로: ").strip().strip('"')).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"폴더를 찾을 수 없습니다: {folder}")

    script_dir = Path(__file__).resolve().parent
    model = YOLO(str(script_dir / "sanjae_20260914_v35.pt"))

    # 모델의 클래스 이름으로 fall ID를 찾습니다.
    fall_ids = {
        class_id for class_id, name in model.names.items()
        if name.strip().lower() == "fall"
    }
    if not fall_ids:
        raise ValueError("모델에 fall 클래스가 없습니다.")

    image_count = 0
    fall_image_count = 0
    total_fall_count = 0

    for result in model.predict(
        source=str(folder),
        save=True,
        stream=True,
        verbose=False,
        project=str(script_dir / "results"),
        name=folder.name,
    ):
        image_count += 1
        # 한 이미지에서 검출된 fall 박스 수를 셉니다.
        fall_count = (
            sum(int(class_id) in fall_ids for class_id in result.boxes.cls.tolist())
            if result.boxes is not None else 0
        )
        if fall_count > 0:
            fall_image_count += 1
            total_fall_count += fall_count
            print(
                f"{Path(result.path).name} | fall {fall_count}개 "
                f"(전체 이미지 {image_count}번째)",
                flush=True,
            )

   
    print(f"fall이 검출된 이미지: {fall_image_count}장")
    print(f"총 fall 검출 수: {total_fall_count}개")
