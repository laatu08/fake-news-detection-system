import React,{useState} from 'react'
import axios from 'axios'


const NewsChecker = () => {
    const [text,setText]=useState("")
    const [result,setResult]=useState(null)
    const [loading,setLoading]=useState(false)
    const [error,setError]=useState(null)


    const checkNews=async()=>{
        if(!text.trim()){
            setError("Please Enter some text to analyse")
            return;
        }

        setLoading(true)
        setError(null)

        try {
            const response=await axios.post("http://localhost:5000/api/predict",{text});
            setResult(response.data);
        } catch (error) {
            setError("Error Displaying News. Please Try Again")
        }
        finally{
            setLoading(false)
        }
    };

  return (
    <div className='container'>
        <h2>📰 Fake News Detector</h2>
        <textarea name="" rows="5" placeholder='Paste a news artical here...' value={text} onChange={(e)=>{setText(e.target.value)}}></textarea>
        <button onClick={checkNews} disabled={loading}>
            {loading?"Checking...":"Check Credibility"}
        </button>

        {error && <p className='error'>{error}</p>}


        {result && (
            <div className="result">
                <h3>Result:</h3>
                <p>
                    <strong>Prediction:</strong> {result.prediction}
                </p>
                <p>
                    <strong>Confidence:</strong> {Math.round(result.confidence*100)}%
                </p>
            </div>
        )}
    </div>
  )
}

export default NewsChecker
