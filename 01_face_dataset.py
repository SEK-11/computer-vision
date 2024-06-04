import cv2
import os

def capture_images(name, num_images=20):
    """Captures images from the webcam and saves them to a person-specific folder.

    Args:
        name (str): Name of the person for whom images are captured.
        num_images (int, optional): Number of images to capture. Defaults to 20.
    """

    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error opening webcam!")
        return
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + '/home/vmukti/Desktop/face detect/haarcascade_frontalface_default.xml')
    try:
        # Create a person-specific folder within "dataset"
        dataset_dir = "dataset"
        person_dir = os.path.join(dataset_dir, name)
        os.makedirs(person_dir, exist_ok=True)  # Safe folder creation
    
        count = 0
        while count < num_images:
            ret, frame = cap.read()

            if not ret:
                print("Error capturing image!")
                break

            cv2.imshow('Capturing...', frame)

            # Capture image on key press (e.g., 'c' for capture)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                image_path = os.path.join(person_dir, f"{name}_{count}.jpg")
                cv2.imwrite(image_path, frame)
                count += 1
                print(f"Image {count} captured!")

            elif key == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Release resources and close windows
        cap.release()
        cv2.destroyAllWindows()

# Example usage
name = input("Enter your name: ")
capture_images(name)
