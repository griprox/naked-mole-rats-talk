from src.audiodata_processing.process_waves import process_waves, butter_highpass_filter, denoise_recording
from src.audiodata_processing.extract_features_from_wave import extract_specs, extract_melspecs
from src.audiodata_processing.process_spectrograms import augment_im, resize_with_padding
from src.metadata_processing.load_data import load_sounds
import numpy as np


def interpret_answers(grouped_probs, all_classes, when_not_sure = 'weirdo'):
    """ Interprets predictions for augmentations of sound.
        If different copies have different predictions when probably clf is not sure """
    result = []
    for gr_p in grouped_probs:
        predicted_classes = np.argmax(gr_p, 1)
        bc = np.bincount(predicted_classes)
        if np.max(bc) >= 3 * len(gr_p) / 5:
            result.append(all_classes[np.argmax(bc)])
        else:
            result.append(when_not_sure)
    return result


def classify_sounds_metadata(model, sounds_metadata, recs_dict, all_classes,
                             all_params_dict, return_grouped=False, times_augment=1):
    """ Classifies sounds from sounds_metadata """

    use_rec_denoising = all_params_dict['rec_denoising']['use_rec_denoising']
    rec_denoising_params = all_params_dict['rec_denoising']['rec_denoising_params']
    sr = all_params_dict['sr']

    recs_dict_to_use = {}
    for name, rec in recs_dict.items():
        if name in sounds_metadata['rec'].unique():
            recs_dict_to_use[name] = denoise_recording(rec, **rec_denoising_params) if use_rec_denoising else rec

    sounds_npy_aug = []
    for i in range(times_augment):
        sounds_npy_aug.extend(load_sounds(sounds_metadata, recs_dict_to_use,
                                          noisy_sampling=True, timestamps='sec'))
    frequency_threshold = all_params_dict['features']['frequency_threshold']
    sounds_npy_aug = np.array([butter_highpass_filter(sound, frequency_threshold, sr)
                               for sound in sounds_npy_aug])

    use_melspecs = all_params_dict['features']['use_melspecs']
    n_fft = all_params_dict['features']['n_fft']
    n_mel = all_params_dict['features']['n_mel']
    target_shape = all_params_dict['features']['target_shape']

    if use_melspecs:
        melspecs = extract_melspecs(sounds_npy_aug, sr, n_fft, n_mel)
    else:
        melspecs = extract_specs(sounds_npy_aug, n_fft)

    melspecs = np.expand_dims(np.array([resize_with_padding(im, target_shape) for im in melspecs]), -1)

    predictions = model.predict(melspecs)
    grouped_probs = [predictions[i::len(sounds_metadata)] for i in range(len(sounds_metadata))]
    if return_grouped:
        return grouped_probs
    predicted_labels = interpret_answers(grouped_probs, all_classes)
    return predicted_labels, melspecs
