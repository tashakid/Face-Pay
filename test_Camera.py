import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera is accessible")
    ret, frame = cap.read()
    if ret:
        print("✅ Can capture frames")
        cv2.imshow('Test', frame)
        cv2.waitKey(3000)  # Show for 3 seconds
    else:
        print("❌ Cannot capture frames")
    cap.release()
else:
    print("❌ Camera not accessible")
cv2.destroyAllWindows()