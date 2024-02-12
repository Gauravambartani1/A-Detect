import React, { useRef, useState } from "react";
import { useSelector } from "react-redux";
import BarChart from "./BarChart";
import LandingPage from "./LandingPage";

const HomePage = () => {
  const { userInfo } = useSelector((state) => state.auth);
  const [loading, setLoading] = useState(false); 

  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };


  const handleClear = (e) => {
    e.preventDefault();
    document.getElementById("mri").value = null;
    setSelectedFile(null);
    setPrediction(null);
  };

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!selectedFile) {
      console.log("Please select an image file");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      setLoading(true);
      const response = await fetch("http://localhost:8080/predict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        
        setTimeout(() => {
          setPrediction(result.class_probabilities);
          setLoading(false);  
        }, 3000);
      } else {
        setLoading(false);
        console.error("Error:", response.statusText);
      }
    } catch (error) {
      setLoading(false);
      console.error("Error:", error.message);
    }
  };

  const maxKey = prediction && Object.keys(prediction).reduce((a, b) => prediction[a] > prediction[b] ? a : b);


  return (
    <div className="flex justify-center items-center flex-col min-h-[var(--min-page-height)] text-4xl font-bold">
      <div className="mb-[var(--navbar-height)] text-center">
        {userInfo ? (
          <>
            <div className="text-lg">
              <form
                onSubmit={handleUpload}
                className="flex flex-col gap-4 items-center"
              >
                <h1 className="text-2xl font-extrabold">
                  Upload your MRI scan
                </h1>
                <input
                  type="file"
                  id="mri"
                  name="mri"
                  className="file-input file-input-primary"
                  accept="image/*"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                {selectedFile && (
                  <img
                    src={URL.createObjectURL(selectedFile)}
                    alt="Selected"
                    style={{
                      maxWidth: "300px",
                      maxHeight: "300px",
                    }}
                  />
                )}
                <div className="flex gap-4">
                  <button
                    onClick={handleClear}
                    className="btn btn-neutral bg-red-500 text-white"
                  >
                    Clear
                  </button>
                  <button
                  type="submit"
                  className="btn-accent btn font-bold w-24"
                  disabled={!selectedFile || loading} // Disable button while loading
                >
                  {loading ? <span className="loading loading-spinner loading-lg"></span> : "Submit"}
                </button>
                </div>
              </form>
              {prediction && (
                <div className="mt-8">
                  <h2>Server Response: {maxKey}</h2>
                  <br />
                  <div className="w-[600px] h-[600px]">
                    <h2>Class Probabilities Chart:</h2>
                    <BarChart data={prediction} className="w-full h-96" />
                  </div>
                </div>
              )}
            </div>
            {/* <div className="">
              <h2>Class Probabilities Chart:</h2>
              <BarChart data={prediction} className="w-full h-96" />
            </div> */}
          </>
        ) : (
          <>
            <LandingPage />
          </>
        )}
      </div>
    </div>
  );
};

export default HomePage;
