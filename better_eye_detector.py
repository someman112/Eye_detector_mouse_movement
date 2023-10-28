import cv2
from mediapipe import solutions
import pyautogui

face_mesh = solutions.face_mesh.FaceMesh(refine_landmarks=True)


def face_detector(img, w, h):
    mg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(mg)
    pts = output.multi_face_landmarks
    image_with_rectangle = None
    b = False
    loi = [474, 475, 476, 477, 13, 14]
    if pts:
        for i in loi:
            x = int(pts[0].landmark[i].x * w)
            y = int(pts[0].landmark[i].y * h)
            new_x = int(pts[0].landmark[i].x * w) - 320
            new_y = int(pts[0].landmark[i].y * h) - 160
            if i == 474:
                screen_size = pyautogui.size()
                rel_x = (new_x / 200) * screen_size[0]
                rel_y = (new_y / 150) * screen_size[1]

                if 0 < rel_x < screen_size[0] and 0 < rel_y < screen_size[1]:
                    b = True
                    pyautogui.moveTo(pyautogui.size()[0] - rel_x, rel_y)
                    image_with_rectangle = cv2.rectangle(img, (300, 150), (500, 300), (0, 255, 0), 2)
                else:
                    image_with_rectangle = cv2.rectangle(img, (300, 150), (500, 300), (0, 0, 255), 2)

                if 0.047 < pts[0].landmark[14].y - pts[0].landmark[13].y < 0.105:
                    pyautogui.click()

            cv2.circle(img, (x, y), 3, (52, 198, 235), -1)

        image_with_rectangle = cv2.flip(image_with_rectangle, 1)
        if b:
            image_with_rectangle = cv2.putText(image_with_rectangle, "In Frame", (140, 330),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            image_with_rectangle = cv2.putText(image_with_rectangle, "Not in Frame", (140, 330),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        img = cv2.rectangle(img, (300, 150), (500, 300), (0, 0, 255), 2)
        img = cv2.flip(img, 1)
        img = cv2.putText(img, "No face detected", (140, 330),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return img

    return image_with_rectangle


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('face detector', face_detector(frame, frame.shape[1], frame.shape[0]))

    if cv2.waitKey(1) == 13:
        break
