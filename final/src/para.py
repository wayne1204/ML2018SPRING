import os

base_dir = os.path.dirname((os.path.realpath(__file__)))
train_dir = os.path.join(base_dir, '../data/audio_train/')
test_dir = os.path.join(base_dir, '../data/audio_test/')
save_dir = os.path.join(base_dir, '../ckpt/')
train_spectrum_dir = os.path.join(base_dir, '../data/train_spectrum/')
train_spectrum_mic_dir = os.path.join(base_dir, '../data/train_spectrum_music/')
train_spectrum_mic_ex_dir = os.path.join(base_dir, '../data/train_spectrum_music_ex/')
test_spectrum_mic_ex_dir = os.path.join(base_dir, '../data/test_spectrum_music_ex/')
test_spectrum_dir = os.path.join(base_dir, '../data/test_spectrum/')
train_enc_dir = os.path.join(base_dir, '../data/train_enc/')
test_enc_dir = os.path.join(base_dir, '../data/test_enc/')
new_csv_dir = os.path.join(base_dir, '../new_csv/')

category = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing" ]

all_class = list(range(41))
music = [0, 3, 6, 7, 8, 11, 12, 14, 18, 19, 20, 22, 23, 29, 30, 33, 35, 38, 39]
nonmusic = [1, 2, 4, 10, 15, 16, 26, 27, 34, 36, 5, 9, 13, 17, 21, 24, 25, 28, 31, 32, 37, 40]

strings = [0, 6, 12, 39]
musical = [7, 8, 18, 22, 29, 30, 38]
percussion = [3, 11, 14, 19, 20, 23, 33, 35]
creature = [1, 2, 4, 10, 15, 16, 26, 27, 34, 36]
non_human = [5, 9, 13, 17, 21, 24, 25, 28, 31, 32, 37, 40]

#human_category = []
# training parameters
max_time = 439
n_fft = 1024
fre_bin = int(n_fft/2 + 1)
num_classes = len(category)
hop_length = int(n_fft/ 1.5)
sr = 44100
epochs = 200
batch_size = 16

params = {'dim': (fre_bin, max_time),
          'batch_size': batch_size,
          'n_classes': num_classes,
          'n_channels': 1,
          'shuffle': True}
