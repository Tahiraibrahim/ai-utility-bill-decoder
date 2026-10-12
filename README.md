# ⚡ AI Utility Bill Decoder

An AI-powered, agentic workflow designed to extract structured data and answer complex visual queries from utility bills. Developed as a submission for the **WomenTechQuest (WTQ) Build Track Hackathon**.

## 🚀 Project Overview
This project leverages **Google Gemini to process images of electricity bills across two main levels:
1. **Level 1 (Data Extraction):** Analyzes raw bill images and extracts critical billing information into a strictly formatted JSON structure, saving it systematically to a CSV file.
2. **Level 2 (Visual Q&A):** Acts as an intelligent customer support agent, answering complex logical and mathematical questions (e.g., tariff comparisons, historical usage differences) based purely on the visual context of the bill.

## ✨ Key Features & Engineering
* **Agentic Workflow:** Fully automated end-to-end Python pipeline requiring zero manual intervention during execution.
* **Robust Error Handling:** Custom implementation of exponential backoff to handle API rate limits (HTTP 429) gracefully without crashing.
* **Progressive Data Saving:** Results are appended incrementally to prevent data loss during large-scale batch processing.
* **Fault Tolerance:** Smartly bypasses missing, duplicate, or corrupted images while allowing the rest of the pipeline to continue seamlessly.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **AI Model:** Gemini 3.5 Flash-Lite (Google Generative AI Vision)
* **Data Handling:** CSV, JSON

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TahiraIbrahim/ai-utility-bill-decoder.git](https://github.com/TahiraIbrahim/ai-utility-bill-decoder.git)
   cd ai-utility-bill-decoder
