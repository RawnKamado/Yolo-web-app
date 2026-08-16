import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (file) {
      setSelectedFile(file);
    }
  };

  const handleStart = () => {
    if (!selectedFile) {
      alert("Please choose an image or video first!");
      return;
    }

    console.log("Selected file:", selectedFile);
  };

  return (
    <div className="app">

      {/* Top control area */}
      <div className="control-row">

        {/* File input */}
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

        {/* Start button */}
        <button
          className="start-button"
          onClick={handleStart}
        >
          START
        </button>

      </div>

      {/* YOLO display area */}
      <div className="yolo-area">

        <div className="yolo-placeholder">
          <div className="target-icon">◎</div>

          <h1>YOLO area</h1>

          <p>
            Uploaded image/video will be displayed here
            <br />
            with YOLO detection results
          </p>
        </div>

      </div>

    </div>
  );
}

export default App;