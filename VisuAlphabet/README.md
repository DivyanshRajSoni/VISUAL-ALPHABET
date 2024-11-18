# VisuAlphabet

This project named as VisuALphabet, It has two major modules - Text to image and Image to text.

# Text to image - 
User can generate image from given text.
# Image to text - 
User can generate description of given image or in short user can caption the image.

# Environment Setup

Create an account on https://huggingface.co/ if you don't have already, and generate and access token.

To generate access token from follow the below steps - 

Log In: Log in to your Hugging Face account.

Go to Your Settings: Click on your profile picture or username in the top right corner, then click on "Settings" from the dropdown menu.

Create a New Token: In the settings, navigate to the "API" section. Here, you'll find a button to create a new API token. Click on it.

Generate Token: Hugging Face will prompt you to create a new API token. Give your token a name and define the scope of the token (for example, read/write access). After configuring your token, click on the "Create" or "Generate" button.

Copy Your Token: Once generated, copy the access token and paste it in auth.py file


Make sure Anaconda is installed and launch anaconda prompt and navigate to root directory in the anaconda prompt

create venv

```shell
conda create -n visualphabet
```

Activate

```shell
conda activate visualphabet
```

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

run the app.py file 

```shell
python app.py
```

Once you see this url - http://127.0.0.1:5000/ in logs, open it in browser.

Now your setup is ready.