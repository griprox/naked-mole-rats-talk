from src.metadata_processing.process_sounds_metadata import generate_sounds_metadata, split_in_train_and_test
from src.metadata_processing.load_data import load_recordings_metadata, load_sounds
from src.metadata_processing.process_sounds_metadata import make_fixed_size_sounds
from src.audiodata_processing.process_waves import process_waves, butter_highpass_filter, denoise_recording
from src.audiodata_processing.extract_features_from_wave import extract_specs, extract_melspecs
from src.audiodata_processing.process_spectrograms import augment_im, resize_with_padding
import pandas as pd
import numpy as np
import pickle
import os


def create_data_for_splitter(all_params_dict, dataset_name, path_to_save, save=False, preloaded_recs_dict=None):
    # Load recordings metadata and recording files. Denoise recordings if needed
    path_to_recordings_metadata = all_params_dict['recs_metadata']['path_to_recordings_metadata']
    recordings_metadata_name = all_params_dict['recs_metadata']['recordings_metadata_name']
    dates = all_params_dict['recs_metadata']['dates']
    colonies = all_params_dict['recs_metadata']['colonies']
    experiments = all_params_dict['recs_metadata']['experiments']
    stages = all_params_dict['recs_metadata']['stages']
    use_rec_denoising = all_params_dict['rec_denoising']['use_rec_denoising']
    rec_denoising_params = all_params_dict['rec_denoising']['rec_denoising_params']
    sr = all_params_dict['sr']

    recs_metadata, recs_dict = load_recordings_metadata(path_to_recordings_metadata, recordings_metadata_name,
                                                        dates, colonies, experiments, stages, preloaded_recs_dict)
    if use_rec_denoising:
        recs_dict = {rec_name: denoise_recording(rec, sr, **rec_denoising_params)
                     for rec_name, rec in recs_dict.items()}
    # Create sounds metadata, split in train and test
    classes_to_drop = all_params_dict['sounds_metadata']['classes_to_drop']
    max_sounds_per_class = all_params_dict['sounds_metadata']['max_sounds_per_class']
    min_sound_length = all_params_dict['sounds_metadata']['min_sound_length']
    max_sound_length = all_params_dict['sounds_metadata']['max_sound_length']
    train_ratio = all_params_dict['sounds_metadata']['train_ratio']
    columns_to_copy = all_params_dict['sounds_metadata']['columns_to_copy']
    resolution = all_params_dict['features']['resolution']
    step = all_params_dict['features']['step']

    sounds_metadata = generate_sounds_metadata(recs_metadata, classes_to_drop, max_sounds_per_class,
                                               min_sound_length, max_sound_length, columns_to_copy,
                                               sr=sr, print_counts=False)
    sounds_metadata_train, sounds_metadata_test = split_in_train_and_test(sounds_metadata, train_ratio=train_ratio)
    sounds_metadata_train = make_fixed_size_sounds(sounds_metadata_train, resolution, step, sr)
    sounds_metadata_test = make_fixed_size_sounds(sounds_metadata_test, resolution, step, sr)
    # Extract sounds from the recordings
    sounds_npy_train = load_sounds(sounds_metadata_train, recs_dict, noisy_sampling=False, timestamps='int')
    sounds_npy_test = load_sounds(sounds_metadata_test, recs_dict, noisy_sampling=False, timestamps='int')
    # Filter waves
    frequency_threshold = all_params_dict['features']['frequency_threshold']
    sounds_npy_train = np.array([butter_highpass_filter(sound, frequency_threshold, sr)
                                 for sound in sounds_npy_train])
    sounds_npy_test = np.array([butter_highpass_filter(sound, frequency_threshold, sr)
                                for sound in sounds_npy_test])
    # Augment waves
    wave_augment_params = all_params_dict['features']['wave_augment_params']
    if wave_augment_params['augment']:
        sounds_npy_train = np.concatenate([sounds_npy_train] * wave_augment_params['times_each_sound'])
        sounds_npy_train = process_waves(sounds_npy_train, wave_augment_params['stretching_lims'],
                                         wave_augment_params['add_noise_lims'], None)
        sounds_metadata_train = pd.concat([sounds_metadata_train] *
                                          wave_augment_params['times_each_sound']).reset_index(drop=True)
    # Extract spectrograms
    use_melspecs = all_params_dict['features']['use_melspecs']
    n_fft = all_params_dict['features']['n_fft']
    n_mel = all_params_dict['features']['n_mel']
    target_shape = all_params_dict['features']['target_shape']
    times_augment_im = all_params_dict['features']['times_augment_im']
    use_augment_im = all_params_dict['features']['augment_im']
    if use_melspecs:
        melspecs_train = extract_melspecs(sounds_npy_train, sr, n_fft, n_mel)
        melspecs_test = extract_melspecs(sounds_npy_test, sr, n_fft, n_mel)
    else:
        melspecs_train = np.array(extract_specs(sounds_npy_train, n_fft))
        melspecs_test = np.array(extract_specs(sounds_npy_test, n_fft))
    # Augment spectrograms
    sounds_npy_train = np.concatenate([sounds_npy_train] * times_augment_im, 0)
    sounds_metadata_train = pd.concat([sounds_metadata_train] * times_augment_im, 0)
    melspecs_train_aug = []
    for _ in range(times_augment_im):
        melspecs_train_aug.extend(melspecs_train)
    if use_augment_im:
        melspecs_train = np.array([augment_im(im, target_shape) for im in melspecs_train_aug])
    else:
        melspecs_train = np.array([resize_with_padding(im, target_shape) for im in melspecs_train_aug])
    del melspecs_train_aug
    melspecs_test = np.array([resize_with_padding(im, target_shape) for im in melspecs_test])

    if save:
        if not os.path.isdir(path_to_save + dataset_name):
            os.makedirs(path_to_save + dataset_name)
        np.save(path_to_save + dataset_name + '/sounds_npy_train.npy', sounds_npy_train)
        np.save(path_to_save + dataset_name + '/sounds_npy_test.npy', sounds_npy_test)
        np.save(path_to_save + dataset_name + '/melspecs_train.npy', melspecs_train)
        np.save(path_to_save + dataset_name + '/melspecs_test.npy', melspecs_test)
        sounds_metadata_train.to_csv(path_to_save + dataset_name + '/sounds_metadata_train.csv', index=False)
        sounds_metadata_test.to_csv(path_to_save + dataset_name + '/sounds_metadata_test.csv', index=False)
        with open(path_to_save + dataset_name + '/params_dict.pickle', 'wb') as f:
            pickle.dump(all_params_dict, f)
        print('Saved !!! ')
    return (sounds_metadata_train, sounds_npy_train, melspecs_train,
            sounds_metadata_test, sounds_npy_test, melspecs_test)


def load_dataset(path_to_dataset, dataset_name):
    sounds_npy_train = None
    sounds_npy_test = None
    melspecs_train = np.load(path_to_dataset + dataset_name + '/melspecs_train.npy')
    melspecs_test = np.load(path_to_dataset + dataset_name + '/melspecs_test.npy')
    sounds_metadata_train = pd.read_csv(path_to_dataset + dataset_name + '/sounds_metadata_train.csv')
    sounds_metadata_test = pd.read_csv(path_to_dataset + dataset_name + '/sounds_metadata_test.csv')
    with open(path_to_dataset + dataset_name + '/params_dict.pickle', 'rb') as f:
        all_params_dict = pickle.load(f)
    return (sounds_metadata_train, sounds_npy_train, melspecs_train,
            sounds_metadata_test, sounds_npy_test, melspecs_test)
