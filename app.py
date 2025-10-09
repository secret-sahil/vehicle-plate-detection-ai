
import cv2
import threading
from flask import Flask, Response, render_template_string, request, jsonify
from inference_pipeline import StreamProcessor, ANPR_RESULTS
from ui import HTML_TEMPLATE
# --- Flask App Initialization ---
app = Flask(__name__)

# Dictionary to hold our stream processor instances, keyed by stream ID
stream_processors = {}
stream_lock = threading.Lock()


# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Starts the ANPR pipeline for a given RTSP stream."""
    data = request.json
    rtsp_url = data.get('rtsp_url')
    print(rtsp_url)
    stream_id = data.get('stream_id', 'default_stream')

    if not rtsp_url:
        return jsonify({"error": "rtsp_url is required"}), 400

    with stream_lock:
        if stream_id in stream_processors and stream_processors[stream_id].is_running():
            return jsonify({"error": f"Stream '{stream_id}' is already running"}), 400
        
        # Create and start a new processor
        processor = StreamProcessor(rtsp_url, stream_id)
        processor.start()
        stream_processors[stream_id] = processor
        
    return jsonify({"message": f"Stream '{stream_id}' started successfully."}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stops a running ANPR pipeline."""
    data = request.json
    stream_id = data.get('stream_id')
    
    with stream_lock:
        if stream_id in stream_processors:
            stream_processors[stream_id].stop()
            del stream_processors[stream_id] # Remove it from the dict
            return jsonify({"message": f"Stream '{stream_id}' stopped."}), 200
        else:
            return jsonify({"error": f"Stream '{stream_id}' not found."}), 404

def generate_frames(stream_id):
    """A generator function that yields frames from the stream processor."""
    while True:
        with stream_lock:
            # Check if the processor still exists and is running
            if stream_id not in stream_processors or not stream_processors[stream_id].is_running():
                print(f"Stopping frame generation for '{stream_id}' as processor is stopped.")
                break
            
            processor = stream_processors[stream_id]

        # Get the latest frame for streaming
        frame = processor.get_display_frame()

        if frame is None:
            # If the pipeline is starting up or has an issue, wait briefly
            cv2.waitKey(100)
            continue
            
        # Encode the frame as JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        
        # Yield the frame in the multipart format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Streams the processed video feed."""
    stream_id = request.args.get('stream_id')
    if not stream_id or stream_id not in stream_processors:
        return "Error: Invalid stream ID.", 404
        
    return Response(generate_frames(stream_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def get_results():
    """Endpoint for the UI to poll for the latest OCR results."""
    stream_id = request.args.get('stream_id')
    if not stream_id:
        return jsonify({"error": "stream_id is required"}), 400
        
    # Access the global results dictionary
    plates = ANPR_RESULTS.get(stream_id, [])
    return jsonify({"plates": list(plates)})


if __name__ == '__main__':
    # It's recommended to run Flask with a production-ready WSGI server like Gunicorn or Waitress
    # For development:
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
