from PIL import Image
from io import BytesIO
from kafka import KafkaConsumer
consumer = KafkaConsumer("StockPrediction",bootstrap_servers=['localhost:29092'],
                        api_version=(0,10,1))

for message in consumer:
    print("Message Received")
    stream = BytesIO(message.value)
    image = Image.open(stream).convert("RGBA")
    print(image.save)
    stream.close()
    image.show()
