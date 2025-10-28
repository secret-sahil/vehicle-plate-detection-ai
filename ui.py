
# --- HTML/JS Frontend ---
# A simple web UI to display the video stream and results
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ANPR Live Stream</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; color: #333; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        h1 { color: #1c1e21; }
        .container { max-width: 960px; width: 100%; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .stream-controls { margin-bottom: 20px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        input[type="text"] { padding: 10px; border: 1px solid #ddd; border-radius: 6px; flex-grow: 1; min-width: 200px; }
        button { padding: 10px 15px; border: none; border-radius: 6px; background-color: #007bff; color: white; font-size: 16px; cursor: pointer; transition: background-color 0.3s; }
        button:hover { background-color: #0056b3; }
        #stop-button { background-color: #dc3545; }
        #stop-button:hover { background-color: #c82333; }
        .stream-view { margin-top: 20px; }
        img { width: 100%; border-radius: 8px; background-color: #eee; }
        h2 { border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 20px; }
        #results-container {
            margin-top: 10px;
            background: #fafafa;
            padding: 15px;
            border-radius: 6px;
            height: 200px;
            overflow-y: auto;
            border: 1px solid #eee;
        }
        .result-item {
            padding: 8px;
            border-bottom: 1px solid #ddd;
            font-family: 'Courier New', Courier, monospace;
            font-size: 16px;
        }
        .status { margin-top: 10px; padding: 10px; background-color: #e9ecef; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-time ANPR Pipeline</h1>
        <div class="stream-controls">
            <input type="text" id="rtsp-url" placeholder="Enter RTSP Stream URL">
            <button onclick="startStream()">Start Stream</button>
            <button id="stop-button" onclick="stopStream()">Stop Stream</button>
        </div>
        <div id="status-message" class="status">Status: Idle</div>

        <div class="stream-view">
            <img id="video-feed" src="" alt="Video stream will appear here.">
        </div>

        <h2>Detected License Plates</h2>
        <div id="results-container"></div>
    </div>

    <script>
        const streamId = "live_stream_1"; // A static ID for this simple example

        function setStatus(message, isError = false) {
            const statusEl = document.getElementById('status-message');
            statusEl.textContent = `Status: ${message}`;
            statusEl.style.color = isError ? '#dc3545' : '#333';
        }

        async function startStream() {
            const rtspUrl = document.getElementById('rtsp-url').value;
            if (!rtspUrl) {
                alert("Please enter an RTSP URL.");
                return;
            }
            setStatus('Starting stream...');
            try {
                const response = await fetch('/start_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rtsp_url: rtspUrl, stream_id: streamId })
                });
                const data = await response.json();
                if (response.ok) {
                    document.getElementById('video-feed').src = `/video_feed?stream_id=${streamId}&t=${new Date().getTime()}`;
                    setStatus(`Processing stream: ${rtspUrl}`);
                } else {
                    throw new Error(data.error || "Failed to start stream.");
                }
            } catch (error) {
                setStatus(`Error: ${error.message}`, true);
            }
        }

        async function stopStream() {
            setStatus('Stopping stream...');
            try {
                const response = await fetch('/stop_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ stream_id: streamId })
                });
                const data = await response.json();
                 if (response.ok) {
                    document.getElementById('video-feed').src = ""; // Clear video feed
                    setStatus('Idle');
                } else {
                    throw new Error(data.error || "Failed to stop stream.");
                }
            } catch (error) {
                setStatus(`Error: ${error.message}`, true);
            }
        }

        // Fetch results periodically
        setInterval(async () => {
            if (document.getElementById('video-feed').src.includes('video_feed')) {
                 try {
                    const response = await fetch(`/results?stream_id=${streamId}`);
                    const data = await response.json();
                    const resultsContainer = document.getElementById('results-container');
                    resultsContainer.innerHTML = ''; // Clear old results
                    data.plates.forEach(plate => {
                        console.log(plate);
                        const item = document.createElement('div');
                        item.className = 'result-item';
                        item.textContent = `[${plate.timestamp}] - ${plate.plate_text}`;
                        resultsContainer.appendChild(item);
                    });
                    // Auto-scroll to the bottom
                    resultsContainer.scrollTop = resultsContainer.scrollHeight;
                } catch (error) {
                    // Fail silently if stream is not running
                }
            }
        }, 2000); // Poll every 2 seconds
    </script>
</body>
</html>
"""