# Python for Engineering - Course Example - Ai PDF Companion

This project was forked from an excellent youtube tutorial linked below, created by Alejandro AO - Software & Ai. He creates fantastic tutorials and I highly recommend checking him out. 👍

If you enjoy this app, come and visit us at [Python For Engineers](https://james-site-4eb3.thinkific.com/courses/your-first-course), where we show you the ropes of how you can use Python to accelerate your everyday engineering workflow building on examples like this one and much, much more.

See the info below from Alejandro's project.

## Langchain Ask PDF (Tutorial)

>You may find the step-by-step video tutorial to build this application [on Youtube](https://youtu.be/wUAUdEw5oxM).

This is a Python application that allows you to load a PDF and ask questions about it using natural language. The application uses a LLM to generate a response about your PDF. The LLM will not answer questions unrelated to the document.

## How it works

The application reads the PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.


## Installation

To install the repository, please clone this repository and install the requirements:

```
pip install -r requirements.txt
```

You will also need to add your OpenAI API key to the `.env` file.

## Usage

To use the application, run the `main.py` file with the streamlit CLI (after having installed streamlit): 

```
streamlit run app.py
```

## Contributing

This repository is for educational purposes only.


