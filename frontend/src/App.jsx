import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ==========================================
  // CHỌN FILE
  // ==========================================

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setError("");
  };

  // ==========================================
  // GỬI FILE ĐẾN BACKEND
  // ==========================================

  const handleStart = async () => {
    // Không có file
    if (!selectedFile) {
      setError("Please choose an image or video first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      let endpoint;

      // Kiểm tra loại file
      if (selectedFile.type.startsWith("image/")) {
        endpoint = `${API_URL}/api/detect/image`;
      } else if (selectedFile.type === "video/mp4") {
        endpoint = `${API_URL}/api/detect/video`;
      } else {
        throw new Error(
          "Only JPG, JPEG, PNG images and MP4 videos are supported."
        );
      }

      // Gửi request
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      // Backend trả lỗi
      if (!response.ok) {
        const errorData = await response.json();

        throw new Error(
          errorData.detail || "Detection failed."
        );
      }

      // Lấy JSON từ backend
      const data = await response.json();

      console.log("Backend result:", data);

      setResult(data);

    } catch (err) {
      console.error(err);

      setError(err.message);

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      {/* =====================================
          TOP CONTROL
      ====================================== */}

      <div className="control-row">

        {/* FILE INPUT */}

        <label className="file-input">

          <span>
            {selectedFile
              ? selectedFile.name
              : "Choose your image or video"}
          </span>

          <input
            type="file"
            accept="image/jpeg,image/png,image/jpg,video/mp4"
            onChange={handleFileChange}
          />

        </label>

        {/* START BUTTON */}

        <button
          className="start-button"
          onClick={handleStart}
          disabled={loading}
        >
          {loading ? "PROCESSING..." : "START"}
        </button>

      </div>


      {/* =====================================
          ERROR
      ====================================== */}

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}


      {/* =====================================
          YOLO AREA
      ====================================== */}

      <div className="yolo-area">

        {/* Chưa có kết quả */}

        {!result && !loading && (
          <div className="yolo-placeholder">

            <div className="target-icon">
              ◎
            </div>

            <h1>
              YOLO area
            </h1>

            <p>
              Uploaded image/video will be displayed here
              <br />
              with YOLO detection results
            </p>

          </div>
        )}


        {/* Đang xử lý */}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>

            <p>
              YOLO is processing your file...
            </p>
          </div>
        )}


        {/* =================================
            IMAGE RESULT
        ================================== */}

        {result && result.image_url && (
          <div className="result-container">

            <img
              src={result.image_url}
              alt="YOLO detection result"
              className="result-image"
            />

            <div className="result-info">

              <h2>
                Detection Result
              </h2>

              <p>
                Objects detected:{" "}
                <strong>
                  {result.count}
                </strong>
              </p>

              <p>
                Processing time:{" "}
                <strong>
                  {result.processing_time}s
                </strong>
              </p>

            </div>

            {/* Detection details */}

            <div className="detections">

              {result.detections.map(
                (detection, index) => (

                  <div
                    className="detection-item"
                    key={index}
                  >

                    <strong>
                      {detection.class}
                    </strong>

                    <span>
                      Confidence:{" "}
                      {(
                        detection.confidence * 100
                      ).toFixed(1)}
                      %
                    </span>

                  </div>

                )
              )}

            </div>

          </div>
        )}


        {/* =================================
            VIDEO RESULT
        ================================== */}

        {result && result.video_url && (
          <div className="result-container">

            <video
              className="result-video"
              src={result.video_url}
              controls
            />

            <div className="result-info">

              <h2>
                Detection Result
              </h2>

              <p>
                Processing time:{" "}
                <strong>
                  {result.processing_time}s
                </strong>
              </p>

            </div>

            <div className="detections">

              {Object.entries(result.counts).map(
                ([className, count]) => (

                  <div
                    className="detection-item"
                    key={className}
                  >

                    <strong>
                      {className}
                    </strong>

                    <span>
                      {count} detections
                    </span>

                  </div>

                )
              )}

            </div>

          </div>
        )}

      </div>

    </div>
  );
}

export default App;