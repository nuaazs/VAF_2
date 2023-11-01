import requests

def request_search(calling_number, file_path):
   url = "http://127.0.0.1:5000/transcribe/file"
   data = {
      'spkid': calling_number,
      'postprocess':1,
      'channel':0,
   }
   files = {'wav_file': open(file_path, 'rb')}
   response = requests.post(url, files=files, data=data)
   print(response.text)
   if response.json().get("code") == 200:
      text = response.json()['transcription']['text']
      print(f"Transcribe success. calling_number:{calling_number}. text:{text}")
   else:
      print(f"Transcribe failed. calling_number:{calling_number}")


request_search("test_spkid", "./test_audio.wav")