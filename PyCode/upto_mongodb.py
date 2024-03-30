from pymongo import MongoClient
from datetime import datetime
from pytz import timezone

# กำหนดข้อมูลเชื่อมต่อ MongoDB
mongo_uri = "mongodb+srv://myAdmin:kasidate01@cluster0.vjo2bfj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
database_name = "SaveImages"
collection_name = "Images"

# สร้าง MongoClient
client = MongoClient(mongo_uri)

# เลือกฐานข้อมูล
database = client[database_name]

# เลือก collection
collection = database[collection_name]

def upload_image(file_path):
    # อ่านไฟล์รูปภาพ
    with open(file_path, 'rb') as image_file:
        # แปลงรูปภาพเป็น binary data
        image_binary = image_file.read()

    # แปลงเวลาเป็น timezone ของประเทศไทย
    thai_timezone = timezone('Asia/Bangkok')
    upload_time = datetime.now(thai_timezone).strftime('%Y-%m-%d')

    # บันทึกรูปภาพและเวลาใน MongoDB
    image_data = {
        "image": image_binary,
        "upload_time": upload_time
    }

    result = collection.insert_one(image_data)
    print(f"Image uploaded successfully. Object ID: {result.inserted_id}")

# เรียกใช้ฟังก์ชัน upload_image แล้วส่งพาธของรูปภาพ
upload_image("D:\Private\Y3Project\python_project\A11.jpg")





# from pymongo.mongo_client import MongoClient
# uri = "mongodb+srv://myadmin:kasidate01@mycluster.puhoukq.mongodb.net/?retryWrites=true&w=majority&appName=myCluster"
# # Create a new client and connect to the server
# client = MongoClient(uri)
# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)