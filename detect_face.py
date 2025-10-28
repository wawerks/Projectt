import cv2

def is_human_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

# Example
img = r"C:\Users\rjhon\Desktop\Projectt\412.jpg"
print("✅ Human face detected!" if is_human_face(img) else "❌ No face found.")
