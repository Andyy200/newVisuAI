from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO
import speech_recognition as sr
from threading import Thread
import time
import pyttsx3
from g4f.client import Client
from g4f.Provider.GeminiPro import GeminiPro
from pydub import AudioSegment


app = Flask(__name__)

weights_path = "yolov8x.pt"


# Ensure weights are downloaded before running the app
def download_yolov8x_weights_if_needed():
   if not os.path.exists(weights_path):
       print(f"yolov8x weights not found. Downloading now...")
       model = YOLO('yolov8x.pt')
       print(f"yolov8x downloaded successfully.")
   else:
       print(f"Using existing yolov8x weights from {weights_path}")


download_yolov8x_weights_if_needed()


model = YOLO(weights_path)

def parse_arguments() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description="YOLOv8 live")
   parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
   parser.add_argument("--horizontal-fov", default=70.0, type=float, help="Horizontal field of view of the webcam in degrees")
   args = parser.parse_args()
   return args


def get_object_color(frame, bbox):
   x1, y1, x2, y2 = bbox
   object_region = frame[int(y1):int(y2), int(x1):int(x2)]
   mean_color = cv2.mean(object_region)[:3]
   return mean_color


def color_to_description(color):
   color = np.array(color)
   if np.all(color < [50, 50, 50]):
       return "very dark"
   elif np.all(color < [100, 100, 100]):
       return "dark"
   elif np.all(color < [150, 150, 150]):
       return "medium"
   elif np.all(color < [200, 200, 200]):
       return "light"
   else:
       return "very light"


def calculate_angle(position, fov, frame_size):
   center = frame_size / 2
   relative_position = position - center
   angle = (relative_position / center) * (fov / 2)
   return angle


def describe_position(center_x, center_y, frame_width, frame_height):
   horizontal_pos = "center"
   vertical_pos = "center"
   if center_x < frame_width / 3:
       horizontal_pos = "left"
   elif center_x > 2 * frame_width / 3:
       horizontal_pos = "right"
   if center_y < frame_height / 3:
       vertical_pos = "top"
   elif center_y > 2 * frame_height / 3:
       vertical_pos = "bottom"
   return f"{vertical_pos} {horizontal_pos}"


def size_description(width, height, frame_width, frame_height):
   object_area = width * height
   frame_area = frame_width * frame_height
   size_ratio = object_area / frame_area
   if size_ratio < 0.05:
       return "small"
   elif size_ratio < 0.2:
       return "medium"
   else:
       return "large"


def extract_data(frame, results, model, h_fov, frame_width, frame_height):
   object_descriptions = []
   class_counts = {}


   for result in results:
       if result.boxes.xyxy.shape[0] == 0:
           continue


       for i in range(result.boxes.xyxy.shape[0]):
           bbox = result.boxes.xyxy[i].cpu().numpy()
           class_id = result.boxes.cls[i].cpu().numpy()
           class_name = model.names[int(class_id)]


           mean_color = get_object_color(frame, bbox)
           color_description = color_to_description(mean_color)
           object_width = bbox[2] - bbox[0]
           object_height = bbox[3] - bbox[1]
           size_desc = size_description(object_width, object_height, frame_width, frame_height)
           center_x = (bbox[0] + bbox[2]) / 2
           center_y = (bbox[1] + bbox[3]) / 2
           h_angle = calculate_angle(center_x, h_fov, frame_width)
           v_angle = calculate_angle(center_y, h_fov * (frame_height / frame_width), frame_height)


           direction = describe_position(center_x, center_y, frame_width, frame_height)
           description = (f"I see a {size_desc} {class_name} at the {direction}. "
                          f"The color of the object is {color_description}. It is positioned at an angle of {h_angle:.2f} degrees horizontally and "
                          f"{v_angle:.2f} degrees vertically.")
           object_descriptions.append(description)


           if class_name in class_counts:
               class_counts[class_name] += 1
           else:
               class_counts[class_name] = 1


   scene_summary = "Here's what I see: " + ", ".join([f"{count} {name}(s)" for name, count in class_counts.items()])
   return object_descriptions, scene_summary


def process_audio():
   r = sr.Recognizer()
   with sr.Microphone() as source:
       audio = r.listen(source)
       said = ""

       try:
           said = r.recognize_google(audio)
           print(said)
       except Exception as e:
           print("Exception: " + str(e))
   return said.lower()

def dir_scene(img_bytes):
   imgClient = Client(
       api_key="AIzaSyD7gGcC_grUkZx5Ww_N1h_RkeHlj95U6RM",
       provider=GeminiPro
   )
   response = imgClient.chat.completions.create(
       model="gemini-1.5-flash",
       messages=[{"role": "user", "content": "I am a blind person that needs to know details of this image with defining features such as specific objects, their color, and distance from my position. You are a helpful agent that will give a vivid and detailed description of the situation. If there are any people and their faces visible try to guess their emotion based on their face. Please provide an educated guess on what actions are happening in this scene as well as a guess on what may happen next."}],
       image=img_bytes
   )

   return response.choices[0].message.content
def find_obj(obj, results, model, cap):
   for result in results:
       for i in range(result.boxes.xyxy.shape[0]):
           class_id = result.boxes.cls[i].cpu().numpy()
           class_name = model.names[int(class_id)]
           if class_name.lower() == obj.lower():
               bbox = result.boxes.xyxy[i].cpu().numpy()
               center_x = (bbox[0] + bbox[2]) / 2
               center_y = (bbox[1] + bbox[3]) / 2
               position_description = describe_position(center_x, center_y, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
               speak(position_description)
   return None


def generate_scene_description(data_log, dir_log):
 global temp
 client = Client()

 scene_description_prompt = (f"Here is a data log of all the objects detected in the past 30 seconds:+ "+"\n".join(data_log) + "\n"
                             f"Here is a scene log of broder details including actions and predections about the scene as well as descriprions of objects in it:+ " + "\n".join(dir_log) + "\n"
                             f" You are a helpful assistant that will take the data log and the scene log and output a breif but descriptive response. Weight the content of the data log 40% and the scene log 60% in your response. I am a blind person that needs to know the basics of the enviroment and what the current scene entails. Please describe the scene in a natural, breif, but all encapsulating manner." + "\n"
                             f"Also, to be less repetetive dont repeat the same information from last 30 seconds. The overall response can be similar but avoid repeating the same information. The response you put from the last 30 seconds is here: \n"+temp)
 response = client.chat.completions.create(
     model="gpt-4o",
     messages=[{"role": "user", "content": scene_description_prompt}]
 )
 temp = response.choices[0].message.content
 return temp
engine = pyttsx3.init()
def speak(text):
   engine.say(text)
   engine.runAndWait()

def gptDirectory(text, results, model, cap, img_bytes):
  client = Client()
  scene_description_prompt = ("the input is: " + text + "\n" + "You are a directory assistant that will follow the following instructions word for word."+"\n"+
                              "If the input is asking to find an object for example(asking where is ___ or help me find ___), ONLY output verbatim exactly as follows: \"find\" + the object that is trying to be found."+"\n"+
                              "If the input is a question (who, what, when, where, why) about the scene for examplem asking about an object or the scene. ONLY output verbatim exactly as follows: \"question\" + the question asked." +"\n"+
                              "If the input states there is an emergency and the input is calling out for help, ONLY output verbatim exactly as follows: \"help\"." +"\n"+
                              "If the input does not match any of the above, be a helpful assistant and try to assist their inquiry."+"\n")
  response = client.chat.completions.create(
     model="gpt-4o",
    messages=[{"role": "user", "content": scene_description_prompt}]
  )
  output = response.choices[0].message.content
  s = output.split(" ")
  for i in s:
       if (i == "find"):
           s.remove("find")
           find_obj(' '.join(s), results, model, cap)
       elif (i == "question"):
           s.remove("question")
           question(' '.join(s), img_bytes)

def question( quest,img_bytes):
   imgClient = Client(
       api_key="AIzaSyD7gGcC_grUkZx5Ww_N1h_RkeHlj95U6RM",
       provider=GeminiPro
   )
   response = imgClient.chat.completions.create(
       model="gemini-1.5-flash",
       messages=[{"role": "user", "content": "I am a blind person that has a question about this image as follows:" + quest}],
       image=img_bytes
   )


   speak(response.choices[0].message.content)

def main():
    speak("Hello I am VisuAI. I ill be your new eyes. If you have any questions, or want me to find an object, or have an emergency, just say hey vision.")
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    h_fov = args.horizontal_fov

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO(weights_path)

    last_data_log_time = time.time()
    last_dir_log_time = time.time()
    last_update_time = time.time()
    data_log = ""
    dir_log = ""
    wake= "hey vision"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, agnostic_nms=True)

        if results:
            object_descriptions, scene_summary = extract_data(frame, results, model, h_fov, frame_width, frame_height)
            detected_objects = "Here is the scene summary:" + scene_summary + \
                               "Here is a more detailed description of the objects mentioned:".join(object_descriptions)

            current_time = time.time()

            if current_time - last_data_log_time >= 1:
                data_log += f"{time.strftime('%H:%M:%S', time.localtime())}: {detected_objects}\n"
                last_data_log_time = current_time

            if current_time - last_dir_log_time >= 10:
                success, img_encoded = cv2.imencode('.jpg', frame)
                if success:
                    img_bytes = img_encoded.tobytes()

                    dir_description = dir_scene(img_bytes)
                    dir_log += f"{time.strftime('%H:%M:%S', time.localtime())}: {dir_description}\n"
                    last_dir_log_time = current_time

            if current_time - last_update_time >= 75:
                scene_description = generate_scene_description(data_log, dir_log)
                speak(scene_description)
                print(scene_description)
                last_update_time = current_time
                data_log = ""
                dir_log = ""

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        speak("speak now")
        text = get_audio()
        

        if text.count(wake) > 0: # we can try to make this async
           speak("I am ready")
           text = get_audio()
           gptDirectory(text, results, model, cap, frame)
           speak("Do you need anything else? If so, say 'Hey Vision'.")


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

@app.route('/')
def index():
   return render_template('index.html')  # Serves the index.html from the templates directory

@app.route('/process_image', methods=['POST'])
def process_image():
   image_file = request.files['image'].read()  
   np_img = np.frombuffer(image_file, np.uint8)  
   img = cv2.imdecode(np_img, cv2.IMREAD_COLOR) 


   results = model(img)
   frame_height, frame_width = img.shape[:2]
   h_fov = 70.0


   # Extract data from the frame and results
   object_descriptions, scene_summary = extract_data(img, results, model, h_fov, frame_width, frame_height)


   # Log detected objects and scene descriptions to the terminal
   print("Scene summary:", scene_summary)
   for description in object_descriptions:
       print(description)


   # Return the scene summary and object descriptions as a JSON response
   return jsonify({"scene_summary": scene_summary, "object_descriptions": object_descriptions})


@app.route('/process_speech', methods=['POST'])
def process_speech():
    user_transcript = request.form.get('transcript')
    
    if not user_transcript:
        return jsonify({"response": "Sorry, I didn't catch that."}), 400

    # Set up camera feed and YOLO processing
    cap = cv2.VideoCapture(0)  # Start capturing from the default webcam
    ret, frame = cap.read()  # Read a frame from the camera

    if ret:  # If a frame was successfully captured
        results = model(frame)  # Run YOLO model on the frame
        
        # Convert the frame to bytes for further processing
        success, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        # Now, pass the actual data to gptDirectory
        ai_response = gptDirectory(user_transcript, results, model, cap, img_bytes)
    else:
        ai_response = "Sorry, I couldn't capture the video feed."

    # Release the camera after use
    cap.release()

    return jsonify({"response": ai_response})

# Audio processing route
@app.route('/get_audio', methods=['POST'])
def get_audio():
    said = ""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files['audio']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    # Convert to WAV format using pydub
    sound = AudioSegment.from_file(audio_path)
    converted_path = "converted_audio.wav"
    sound.export(converted_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(converted_path) as source:
        audio_data = recognizer.record(source)

        try:
            transcript = recognizer.recognize_google(audio_data)
            said = transcript
            return said.lower()
        except sr.UnknownValueError:
            return jsonify({"transcript": "Sorry, I couldn't understand the audio."})
        except sr.RequestError as e:
            return jsonify({"transcript": f"Could not request results; {e}"})

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=8001)





