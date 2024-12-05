from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import openai
from io import BytesIO
from faster_whisper import WhisperModel
import logging


# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Flask app setup
app = Flask(__name__)

# OpenAI API Key Setup
openai.api_key = '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'
# Path to emotion recognition model
MODEL_PATH = r'./templates/model/FINALFACEMODEL.keras'
#test-syntax
# Load emotion recognition model
emotion_model = load_model(MODEL_PATH)
emotion_labels = ['angry', 'disgust', 'sad', 'happy', 'neutral', 'fear', 'surprise']

# Function to map emotions to valence and arousal
def get_valence_arousal(emotion):
    emotion_map = {
        'happy': (0.8, 0.7),
        'sad': (0.2, 0.3),
        'angry': (0.1, 0.5),
        'surprise': (0.7, 0.6),
        'fear': (0.2, 0.5),
        'disgust': (0.1, 0.2),
        'neutral': (0.5, 0.5),
    }
    return emotion_map.get(emotion, (0.5, 0.5))

# Analyze emotions from a frame
def analyze_emotion(frame, model):
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = img_to_array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions[0]

# Generate frames for the live feed
def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Largest face by area
            (x, y, w, h) = largest_face
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = frame[y:y + h, x:x + w]
            emotion_data = analyze_emotion(face_roi, emotion_model)
            
            if emotion_data is not None:
                dominant_emotion = np.argmax(emotion_data)
                emotion = emotion_labels[dominant_emotion]
                emotion_score = float(emotion_data[dominant_emotion])
                valence, arousal = get_valence_arousal(emotion)
                
                # Display emotion, valence, and arousal on frame
                cv2.putText(frame, f'Emotion: {emotion} ({emotion_score:.2f})', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Valence: {valence:.2f}, Arousal: {arousal:.2f}', 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes for Flask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/data_privacy')
def data_privacy():
    return render_template('data privacy.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "I didn't catch that. Can you say it again?"})
    
    try:
        # Generate chatbot response using OpenAI API
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}],
        )
        bot_reply = completion.choices[0].message["content"]
        return jsonify({"response": bot_reply})
    except Exception as e:
        return jsonify({"response": f"Error generating response: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)


#Speech to text communication
@app.route('/whisper', methods=['POST'])
def whisper():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio']
        
        # Send the audio file to OpenAI's Whisper API
        transcription_result = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
        )
        transcription = transcription_result.get("text", "")

        return jsonify({"transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)})
    
# Initialize the Whisper model
model = WhisperModel("distil-large-v3", device="cpu", compute_type="float16")

# Transcription function that accepts in-memory audio data
def transcribe_audio(audio_data):
    # Process the audio and transcribe it
    segments, _ = model.transcribe(audio_data, beam_size=5, language="en")
    
    # Combine all the transcribed segments into one text
    transcription = " ".join([segment.text for segment in segments])
    return transcription

# Route for receiving and transcribing the audio data
@app.route('/whisper', methods=['POST'])
def whisper():
    try:
        # Get the audio data from the request
        audio_file = request.files['audio']

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        # Convert the uploaded file to a BytesIO stream
        audio_data = BytesIO(audio_file.read())

        # Transcribe the audio in-memory
        transcription = transcribe_audio(audio_data)

        # Return the transcription result as a JSON response
        return jsonify({"transcription": transcription})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
    
# Data storage
emotion_results = []
chat_history = []

# Helper: Map emotions to valence and arousal
def get_valence_arousal(emotion):
    emotion_map = {
        'happy': (0.8, 0.7),
        'sad': (0.2, 0.3),
        'angry': (0.1, 0.5),
        'surprise': (0.7, 0.6),
        'fear': (0.2, 0.5),
        'disgust': (0.1, 0.2),
        'neutral': (0.5, 0.5),
    }
    return emotion_map.get(emotion, (0.5, 0.5))

# Analyze emotions from a frame
def analyze_emotion(frame, model):
    resized_frame = cv2.resize(frame, (224, 224))
    img_array = img_to_array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions[0]

# Generate frames for the live feed
def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Largest face by area
            (x, y, w, h) = largest_face
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = frame[y:y + h, x:x + w]
            emotion_data = analyze_emotion(face_roi, emotion_model)
            
            if emotion_data is not None:
                dominant_emotion = np.argmax(emotion_data)
                emotion = emotion_labels[dominant_emotion]
                emotion_score = float(emotion_data[dominant_emotion])
                valence, arousal = get_valence_arousal(emotion)
                
                # Display emotion, valence, and arousal on frame
                cv2.putText(frame, f'Emotion: {emotion} ({emotion_score:.2f})', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Valence: {valence:.2f}, Arousal: {arousal:.2f}', 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Store emotion result
                emotion_results.append({
                    'timestamp': datetime.now().isoformat(),
                    'emotion': emotion,
                    'valence': valence,
                    'arousal': arousal
                })
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes for Flask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/results', methods=['GET'])
def video_feed_results():
    # Check if there are any emotion results
    if not emotion_results:
        return jsonify({"videoResults": "No video feed data available yet"})
    
    # Return the emotion results as a JSON response
    return jsonify({"videoResults": emotion_results})


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "I didn't catch that. Can you say it again?"})
    
    try:
        # Generate chatbot response using OpenAI API
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}],
        )
        bot_reply = completion.choices[0].message["content"]
        
        # Simulated emotion classification
        classified_emotion = "neutral"  # Replace with actual classification logic
        
        # Store chat history
        chat_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': user_input,
            'bot_reply': bot_reply,
            'classified_emotion': classified_emotion
        })
        
        return jsonify({"response": bot_reply})
    except Exception as e:
        return jsonify({"response": f"Error generating response: {str(e)}"})

@app.route('/download_data', methods=['GET'])
def download_data():
    today_date = request.args.get("date", datetime.now().strftime('%Y-%m-%d'))
    
    # Prepare the text file content
    content = f"Interaction Data for {today_date}\n\n"
    content += "Emotion Detection Results:\n"
    for result in emotion_results:
        content += f"{result['timestamp']}: {result['emotion']} (Valence: {result['valence']}, Arousal: {result['arousal']})\n"
    
    content += "\nChat History:\n"
    for chat in chat_history:
        content += f"[{chat['timestamp']}]\n"
        content += f"User: {chat['user_message']}\n"
        content += f"Bot: {chat['bot_reply']} (Emotion: {chat['classified_emotion']})\n\n"
    
    # Save to file
    file_path = f"interaction_{today_date}.txt"
    with open(file_path, "w") as f:
        f.write(content)
    
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run()
