import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
from collections import namedtuple
from pymongo import MongoClient
from datetime import datetime
from pytz import timezone
import base64

# กำหนดข้อมูลเชื่อมต่อ MongoDB
mongo_uri = "mongodb+srv://myadmin:kasidate01@mycluster.puhoukq.mongodb.net/?retryWrites=true&w=majority&appName=myCluster"
database_name = "SaveImages"
collection_name = "Images"
# สร้าง MongoClient
client = MongoClient(mongo_uri)
database = client[database_name]
collection = database[collection_name]

cooldown_period = 10  # 1 minute

ZONE_POLYGON = np.array([
    [0.3, 0.05],
    [0.7, 0.05],
    [0.7, 0.95],
    [0.3, 0.95]
])

ModelConfig = namedtuple('ModelConfig', ['path', 'names'])
model_config = ModelConfig(
    path="D:\Private\Y3Project\python_project\Weights\w2024-03-26\W640\wbest640.pt",
    names=["No Helmet", "Person", "Rider", "Wear a helmet"]  # Replace with actual class names
)

rstp_url = 'rtsp://admin:kasidate01@192.168.123.71:554/Streaming/Channels/101'


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def save_image_to_mongodb(image_binary, count_no_helmet, count_rider):
    # Convert time to Thailand timezone
    thai_timezone = timezone('Asia/Bangkok')
    upload_time = datetime.now(thai_timezone).strftime('%Y-%m-%d')

    # Save image and time in MongoDB
    image_data = {
        "image": base64.b64encode(image_binary).decode('utf-8'),
        "upload_time": upload_time,
        "count_no_helmet": count_no_helmet,
        "count_rider": count_rider
    }

    result = collection.insert_one(image_data)
    print(f"Image uploaded successfully. Object ID: {result.inserted_id}")


def main():
    last_save_time = time.time()
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    model = YOLO(model_config.path)
    cap = cv2.VideoCapture(rstp_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    #     box_annotator = sv.BoxAnnotator(
    #         thickness=2,
    #         text_thickness=2,
    #         text_scale=1
    #     )
    box_annotator = sv.RoundBoxAnnotator(
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Initialize FPS display text
    fps_text = "FPS: calculating..."

    fps_start_time = time.time()
    fps_counter = 0

    skip_frames = 5  # Process every 5th frame
    frame_count = 0
    max_retries = 30  # Maximum number of retries
    retry_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            retry_count += 1
            if retry_count <= max_retries:
                print(f"Error reading frame, retrying ({retry_count}/{max_retries})")
                time.sleep(0.15)
                continue
            else:
                print("Max retries reached, exiting.")
                break

        retry_count = 0

        frame_count += 1

        if frame_count % skip_frames != 0:
            continue

        fps_counter += 1
        bg_color = (0, 0, 0)  # Black background
        bgG_color = (0,255,0)
        text_color = (255, 255, 255)  # White text

        for result in model.track(source=frame, stream=True, persist=True):
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            labels = []
            if detections.tracker_id is not None:
                labels = [
                    f"#{tracker_id} {model_config.names[class_id]} {confidence:0.2f}"
                    for class_id, confidence, tracker_id
                    in zip(detections.class_id, detections.confidence, detections.tracker_id)
                ]

            #             frame = box_annotator.annotate(
            #                     scene=frame.copy(),
            #                     detections=detections
            #                 )
            if labels:
                frame = box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections
                )
                frame = label_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections,
                    labels=labels
                )
            else:
                print("No Labels")

            mask = zone.trigger(detections=detections)

            # ตรวจสอบว่า detections มีขนาดไม่เท่ากับ 0
            if len(detections) > 0:
                count_no_helmet = np.count_nonzero((detections.class_id == 0) & (detections.confidence >= 0.5) & mask)
                count_rider = np.count_nonzero((detections.class_id == 2) & (detections.confidence >= 0.5) & mask)

                # Print ค่าจำนวนออกมา
                print(f"จำนวนคนไม่สวมหมวก: {count_no_helmet}")
                print(f"จำนวนคน Rider: {count_rider}")

                if count_no_helmet >= 1 and count_rider >= 1:
                    current_time = time.time()
                    if current_time - last_save_time >= cooldown_period:
                        image_binary = cv2.imencode('.jpg', frame)[1].tobytes()
                        save_image_to_mongodb(image_binary, count_no_helmet, count_rider)
                        print("Save Images Successfully")
                        last_save_time = current_time
            else:
                count_no_helmet = 0
                count_rider = 0

            frame = zone_annotator.annotate(scene=frame.copy())

        # Update FPS text
        fps_counter += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            fps_text = f"FPS: {fps:.2f}"

        cv2.rectangle(frame, (10, 10), (200, 50), bg_color, -1)  # Rounded edge square background
        cv2.putText(frame, fps_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)  # White text

        cv2.imshow("yolov8", frame)
        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
