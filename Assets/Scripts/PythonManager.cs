using UnityEditor.Scripting.Python;
using UnityEditor;
using UnityEngine;
using System.IO;
namespace PythonScripts
{
public class PythonManager
{
[MenuItem("Python/HelloWorld")]
    public static void RunHelloWorld()
    {
        PythonRunner.RunFile($"{Application.dataPath}/Scripts/Python Script/HelloWorld.py");
    }
#region Audio Segmentation
[MenuItem("Python/Part1")]
    public static void RunPart1()
    {
                PythonRunner.RunString(@"
import librosa
import librosa.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random
import string
import os

from pydub import AudioSegment

directory_path = ""E:\\Projects 2023-2024\\VR Project\\Speech Test\\Assets\\Audios""
# List to store the loaded data
data = []

# Iterate through files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith("".wav""):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # Load the WAV file
        wav_data, sampling_rate = librosa.load(file_path)

        # Append the loaded data to the list
        data.append((wav_data, sampling_rate, filename))

# Now 'data' is a list containing tuples of (wav_data, sampling_rate, filename)
# You can iterate through the list and process each entry
for wav_data, sampling_rate, filename in data:
    # Your processing code here
    print(f""Processing {filename}: Data shape: {wav_data.shape}, Sampling rate: {sampling_rate}"")

def split_and_save_audio(input_path, output_folder, chunk_length):
    # Reading audio from the input path
    sound = AudioSegment.from_wav(input_path)
    
    # Calculate the total duration of the audio
    total_duration = len(sound)
    
    # Calculate the number of chunks needed
    num_chunks = total_duration // chunk_length
    
    # Ensure the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Splitting audio into chunks
    for i in range(num_chunks):
        start_time = i * chunk_length
        end_time = (i + 1) * chunk_length
        chunk = sound[start_time:end_time]
        
        # Save each chunk to the output folder
        chunk.export(os.path.join(output_folder, f""chunk_{i+1}.wav""), format=""wav"")
        
    if num_chunks == 0:
        sound.export(os.path.join(output_folder, f""chunk_1.wav""), format=""wav"")

for filename in os.listdir(directory_path):
    input_audio_path = f""E:\\Projects 2023-2024\\VR Project\\Speech Test\\Assets\\Audios\\{filename}""
    if filename.endswith("".wav""):
        
        output_folder_path = f""E:\\Projects 2023-2024\\VR Project\\Speech Test\\Assets\\Interview_Questions\\question_number_{filename}""

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            print(f""Folder '{output_folder_path}' created."")
        else:
            print(f""Folder '{output_folder_path}' already exists."")

        chunk_length_seconds = 5  # You can adjust this value

        split_and_save_audio(input_audio_path, output_folder_path, chunk_length_seconds * 1000)
");
            }
#endregion
#region Sppech Emotion Detection Model
[MenuItem("Python/Part2")]
    public static void RunPart2()
    {
                // path to Interview_Questions directory
        string mainDirectoryPath = Path.Combine(Application.dataPath, "Interview_Questions");;

        // Get the subdirectories (questions of interview)
        string[] subdirectories = Directory.GetDirectories(mainDirectoryPath);
        
        foreach (string subdirectory in subdirectories)
        {
            //Debug.Log("Subdirectory: " + subdirectory);

            // Get the files inside each subdirectory
            string[] files = Directory.GetFiles(subdirectory);

            // Iterate through each file
            foreach (string file in files)
            {
                if (!file.EndsWith(".meta"))
                {
                
                string pythonCompatiblePath = file.Replace("/", "\\\\");
                Debug.Log("Treating File: " + pythonCompatiblePath);
                PythonRunner.RunString($@"
import librosa
import librosa.display

import pandas as pd
import numpy as np

import joblib
from joblib import load

import os
import UnityEngine;

# Extract the directory path without the file name
directory_path = os.path.dirname(""{pythonCompatiblePath}"")

# Normalize the directory path
data_path = os.path.normpath(""{pythonCompatiblePath}"")
#UnityEngine.Debug.Log('printing file from python: {pythonCompatiblePath}')
directory = os.listdir(data_path)
file_path = []

for file in directory:
    file_path.append(data_path + '\\' + file)

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])

data_path = pd.concat([path_df])

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# taking any example and checking for techniques.
path = np.array(data_path.Path)[1]
UnityEngine.Debug.Log('printing file from python :'+path)
data, sample_rate = librosa.load(path)

def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result

X = []

for path in data_path.Path:
    feature = get_features(path)
    for ele in feature:
        X.append(ele)

scaler = load(""F:\\data-for-training\\Emotion_Detection_Model\\scaler.joblib"")
X = scaler.transform(X)

X = np.expand_dims(X, axis=2)

# import trained model
test_model = joblib.load(""F:\\data-for-training\\Emotion_Detection_Model\\model_filename.joblib"")

# prediction
encoder = joblib.load(""F:\\data-for-training\\Emotion_Detection_Model\\encoder.joblib"")
pred_test = test_model.predict(X)
y_pred = encoder.inverse_transform(pred_test)
print(y_pred.shape)

df = pd.DataFrame(columns=['Predictions'])
df['Predictions'] = y_pred.flatten()
df.head()

import statistics

# Assuming df['Predictions'] is your Pandas DataFrame column
predictions = df['Predictions']

# Use the mode function to get the most frequent element(s)
most_frequent = statistics.multimode(predictions)

# Convert the result to an array
result_array = np.array(most_frequent)

# Print or use the result_array as needed
UnityEngine.Debug.Log(""Most Frequent Predictions:"", result_array)");
        }}
    }
    }
#endregion
}
}
