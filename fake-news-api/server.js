const express=require('express')
const cors=require('cors')
const bodyParser=require('body-parser')
const predictRoute=require('./routes/predict.js')


const app=express()
app.use(cors())
app.use(bodyParser.json())


app.use('/api/predict',predictRoute)


const PORT=5000
app.listen(PORT,()=>{
    console.log(`Server is running on port ${PORT}`);
})