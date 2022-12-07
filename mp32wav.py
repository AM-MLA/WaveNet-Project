import os

# to get in parameters
mp3_path = "../smaller_set_mp3/audio/"

mp3_file = os.listdir(mp3_path)

for i, mp3 in enumerate(mp3_file):
    # warn the user about ffmpeg not in path
    os.system("ffmpeg -i "+ mp3_path + mp3
    + " " + mp3_path + str(i) + ".wav" )