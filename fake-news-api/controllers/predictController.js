const { spawn } = require("child_process");
const PYTHON_PATH = "C:/Code/Fake News Detection System/venv/Scripts/python.exe";

exports.predictNews = (req, res) => {
    const { text } = req.body;
    if (!text) {
        return res.status(400).json({ error: "News text is required!" });
    }

    const pythonProcess = spawn(PYTHON_PATH, ["model/predict.py", text]);

    let responseData = "";

    pythonProcess.stdout.on("data", (data) => {
        responseData += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        console.error(`Error: ${data}`);
        res.status(500).json({ error: "Prediction failed" });
    });

    pythonProcess.on("close", () => {
        // Split response into prediction and confidence values
        const lines = responseData.trim().split("\n");
        if (lines.length >= 2) {
            const prediction = lines[0].replace("Prediction: ", "").trim();
            const confidence = parseFloat(lines[1].replace("Confidence: ", "").trim());
            res.json({ prediction, confidence });
        } else {
            res.status(500).json({ error: "Invalid response from model" });
        }
    });
};