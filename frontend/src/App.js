import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import Navbar from './components/navbar';

function App() {
  const [file, setFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [drawnImage, setDrawnImage] = useState(null);
  const [ matrix, setMatrix ] = useState(null);
  const [id, setId] = useState(null);
  const [finalImage, setFinalImage] = useState(null);
  const [time,setTime] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setProcessedImage(null);
      setDrawnImage(null);
      setMatrix(null);
      setFinalImage(null);
      setId(null);
      setTime(null);

      const response = await axios.post('http://localhost:5000/process_image', formData);

      const { processed_image } = response.data;

      // Convert hexadecimal data to base64
      const base64Image = arrayBufferToBase64(hexToUint8Array(processed_image));
      setProcessedImage(`data:image/png;base64,${base64Image}`);
      setMatrix(response.data.matrix);
      setDrawnImage(`data:image/png;base64,${arrayBufferToBase64(hexToUint8Array(response.data.drawn))}`);
      setId(response.data.id);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('An error occurred while uploading the file');
    }
  };
  const handleInputChange = (rowIndex, colIndex, value) => {
    if (/^\d{0,2}$/.test(value)) {
      const newMatrix = [...matrix];
      newMatrix[rowIndex][colIndex] = Number(value);
      setMatrix(newMatrix);
    }
  };

  const Solve = async () => {
    if (matrix && id){
      try{
      const response = await axios.post('http://localhost:5000/solve', {id, matrix});
      const { processed_image } = response.data;
      const base64Image = arrayBufferToBase64(hexToUint8Array(processed_image));
      setFinalImage(`data:image/png;base64,${base64Image}`);
      const timeRegex = /Sudoku solved in (\d+\.\d+) seconds/;
      const match = response.data.output.match(timeRegex);
      setTime(match[0]);
      }catch (error) {
        console.error('Error uploading file:', error);
        alert('An error occurred solving the puzzle');
      }
    }
  }


  const hexToUint8Array = (hexString) => {
    const buffer = new Uint8Array(hexString.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
    return buffer;
  };

  const arrayBufferToBase64 = (buffer) => {
    let binary = '';
    buffer.forEach((byte) => {
      binary += String.fromCharCode(byte);
    });
    return btoa(binary);
  };
  const tablePadding = '10px';

  return (
    <div className="App">
      <Navbar/>
      <input
  type="file"
  onChange={handleFileChange}
  style={{

    padding: '10px',
    border: '1px solid #ccc',
    borderRadius: '5px',
    marginRight: '10px',
  }}
/>
<button
  onClick={handleUpload}
  style={{
    padding: '10px',
    backgroundColor: '#4CAF50',
    color: '#fff',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
  }}
>
  Upload and Process
</button>

      <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'center' }}>
        <div>
          {processedImage && (
            <div style={{
              padding: tablePadding,
            }}>
              <img src={processedImage} alt="Processed" />
            </div>
          )}
        </div>
        <div>
          {drawnImage && (
            <div style={{
              padding: tablePadding,
            }}>
              <img src={drawnImage} alt="drawn" />
            </div>
          )}
        </div>



    </div>
    <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'center' }}>
    {matrix && (
 <table style={{ width: '400px', height: '400px', borderCollapse: 'collapse', margin:"10px"}}>
 <tbody>
   {matrix.map((row, rowIndex) => (
     <tr key={rowIndex}>
       {row.map((value, colIndex) => (
         <td key={colIndex} style={{ width: `${400 / matrix[0].length}px`, height: `${400 / matrix[0].length}px`, border: '1px solid black' }}>
           <input
             type="text"
             value={value}
             onChange={(e) => handleInputChange(rowIndex, colIndex, e.target.value)}
             maxLength={2}
             style={{ width: '100%', height: '100%', boxSizing: 'border-box', border: 'none' }}
           />
         </td>
       ))}
     </tr>
   ))}
 </tbody>
</table>
      )}

{finalImage && (
      <div style={{
        padding: tablePadding,
      }}>
        <img src={finalImage} alt="final" />
      </div>
    )}
      
    </div>
    <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'center' }}>
<div style={{
        padding: '10px',
      }}>
      {matrix && (
        <button style={{
          padding: '10px',
          border: 'none',
          width: '400px',
          fontSize: '16px',
          borderRadius: '5px',
          backgroundColor: 'green',
          color: 'white',
          cursor: 'pointer',
        }}onClick={async ()=>{
          console.log(matrix);
          await Solve()}}>Solve</button>
      )}
      </div>
    </div>
    {time && (
      <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'center' }}>
      <div style={{
        padding: '10px',
      }}>
        <h2>{time}</h2>
      </div>
      </div>
    )}
    </div>
  );
}

export default App;
