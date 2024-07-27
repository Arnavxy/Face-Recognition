import imageio # type: ignore
import cv2

def test_webcam():
    try:
        reader = imageio.get_reader('<video0>')
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    for i, frame in enumerate(reader):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Webcam Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if i == 100:  # Run for a limited number of frames
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()
