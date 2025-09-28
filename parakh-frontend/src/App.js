import React, { useState, useCallback ,useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { FaDropbox } from "react-icons/fa";

function App() {
  const [result, setResult] = useState(null);
  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    if (!file) return;
    // Prepare form data for upload
    const formData = new FormData();
    formData.append("file", file);
    // Send POST request to Flask backend
    axios.post("http://localhost:5000/detect", formData, {
      headers: { "Content-Type": "multipart/form-data" }
    })
      .then(response => {
        // Expecting JSON: { type, size, image }
        setResult(response.data);
      })
      .catch(error => console.error("Upload error:", error));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: 'image/*'
  });

  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/data")
      .then((res) => res.json())
      .then((result) => setData(result))
      .catch((err) => console.error("Error:", err));
  }, []);


  return (
    <div className='bg-blue-200 h-screen flex flex-col items-center justify-center  '>
      <h1 className='text-4xl mt-5 font-bold mb-10 text-center'>PARAKH <br />
        <span className='text-blue-500 font-bold'> Microplastic Detector</span></h1>
      <div {...getRootProps()} className='border-2 border-dashed border-gray-400 p-10 text-center flex flex-col items-center'>
        <input  className='px-10'{...getInputProps()} />
        {
          isDragActive ?
          
            <p className='text-xl font-bold'>Drop the image here ...</p> :(
            <>
            <FaDropbox className='text-4xl mb-4'/>
            <p className='text-xl font-bold'>Drag & drop an image here, or click to select</p>
            </>
            )
        }
      </div>
      {result && (
        <div className='mt-4'>
          <h2>Detection Result:</h2>
          <p>Type: {result.type}</p>
          <p>Size: {result.size}</p>
          {result.image &&
            <img src={`data:image/png;base64,${result.image}`} alt="Processed"
              className='max-w-full border border-gray-400' />}
        </div>
      )}
    </div>
  );
}

export default App;


// import { useEffect, useState } from "react";

// function App() {

//   return (
//     <div>
//       <h1>React + Flask Connected 🎉</h1>
//       <pre>{JSON.stringify(data, null, 2)}</pre>
//     </div>
//   );
// }

// export default App;
