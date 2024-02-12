import React from "react";

const LandingPage = () => {
  return (
    <>
      {/* First Part */}
      <div className="flex h-screen">
        <div className="flex-1 flex items-center justify-center flex-col px-20">
          <h1 className="text-5xl font-bold mb-8 leading-tight bg-clip-text">
            Alzheimer's Detection: A Machine Learning Approach
          </h1>
          <p className="font-light text-lg ">
            Alzheimer's is the leading cause of dementia worldwide, and our aim is to revolutionize early diagnosis through advanced machine learning. Our innovative model analyzes MRI Scans to predict potential risk, paving the way for personalized diagnosis and improved patient care.
          </p>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <img
            className="object-cover h-3/4 w-3/4 rounded-lg my-4"
            src="../../Assets/brain.jpg"
            alt="Brain"
          />
        </div>
      </div>

      {/* Second Part */}
      <div className="flex h-screen">
        <div className=" flex-1 flex items-center justify-center">
          <img
            className="object-cover rounded-xl my-4"
            src="../../Assets/symptoms.jpg"
            alt="Brain"
          />
        </div>
        <div className="flex-1 flex items-center justify-center flex-col px-20">
          <h1 className="text-5xl font-bold mb-8 leading-tight bg-clip-text">
            Understanding the Signs: Early Symptoms of Alzheimer's Disease
          </h1>
          <ul className="list-disc font-light text-lg pl-6">
            <li>Memory loss disrupting daily life</li>
            <li>Challenges in planning or problem-solving</li>
            <li>Difficulty completing familiar tasks</li>
            <li>Confusion with time or place</li>
            <li>Trouble understanding visual images</li>
            <li>Changes in mood and personality</li>
          </ul>
        </div>
      </div>

      {/* Third Part */}
      <div className="flex h-screen">
        <div className="flex flex-col px-20 text-center justify-center">
          <h2 className="text-4xl font-bold mb-8 leading-tight bg-clip-text">
            Unveiling the Algorithm: Inside the Alzheimer's Prediction Model
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-12">
            <div className="text-lg font-normal">
              <p>
                Our model leverages the power of Convolutional Neural Networks (CNNs), inspired by the human brain's structure. By analyzing MRI scans, the model learns to identify subtle patterns and differences that may indicate potential Alzheimer's risk.
              </p>
              <br></br>
              <p>
                Think of it like training a highly specialized doctor who has seen countless MRI scans. Over time, the model develops the ability to detect even the faintest signs of the disease, paving the way for earlier intervention and better outcomes.
              </p>
            </div>
            <div className="flex justify-center">
              <img
                src="../../Assets/model.jpg"
                alt="CNN model diagram"
                className="rounded-lg w-full md:w-2/3 my-4 h-fit"
              />
            </div>
          </div>
          <p className="font-light text-lg mb-8">
            **Important Note:** This model is still under development and intended for research purposes only. It is not a diagnostic tool and should not be used to replace professional medical advice.
          </p>
        </div>
      </div>
    </>
  );
};

export default LandingPage;
