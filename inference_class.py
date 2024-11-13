import cv2
import mediapipe as mp
import pickle
import numpy as np
model_dict=pickle.load(open('./model.p','rb'))
model=model_dict['model']

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawingq_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_dict={0:'A',1:'B',2:'C'}

while True:

    data_aux=[]
    x_=[]
    y_=[]


    ret, frame = cap.read()
    if not ret:
        break
    
    H,W,_=frame.shape

    # Convert the frame to RGB as MediaPipe processes RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:  # Correct spelling here
            # Draw hand landmarks and connections on the original BGR frame
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

        x1=int(min(x_)*W)
        y1=int(min(y_)*H)
        x2=int(max(x_)*W)
        y2=int(max(y_)*H)

        prediction=model.predict([np.asarray(data_aux)])

        predicted_character=labels_dict[int(prediction[0])]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame,predicted_character,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),3,cv2.LINE_AA)


    # Display the frame with drawn landmarks
    cv2.imshow('frame', frame)

    # Add a condition to break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
