from src.audiodata_processing.process_waves import process_waves, butter_highpass_filter, denoise_recording
from src.audiodata_processing.extract_features_from_wave import extract_specs_new, extract_melspecs
from src.audiodata_processing.process_spectrograms import augment_im, resize_with_padding
from src.metadata_processing.load_data import load_sounds
import pandas as pd
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
    sounds_metadata_aug = []
    for i in range(times_augment):
        sounds_npy_aug.extend(load_sounds(sounds_metadata, recs_dict_to_use,
                                          noisy_sampling=True, timestamps='sec'))
        sounds_metadata_aug.append(sounds_metadata)
    sounds_metadata_aug = pd.concat(sounds_metadata_aug).reset_index(drop=True)

    highpass_filtering = all_params_dict['features']['highpass_filtering']
    sounds_npy_aug = np.array([butter_highpass_filter(sound, highpass_filtering, sr) for sound in sounds_npy_aug])

    specs_type = all_params_dict['features']['specs_type']
    extract_specs_params = all_params_dict['features']['extract_specs_params']
    n_fft = extract_specs_params['nperseg']
    n_mel = extract_specs_params['num_freq_bins']
    target_shape = all_params_dict['features']['target_shape']

    if specs_type == 'mel':
        specs = extract_melspecs(sounds_npy_aug, sr, n_fft, n_mel)
    elif specs_type == 'new-specs':
        specs = np.array(extract_specs_new(sounds_metadata_aug, sounds_npy_aug, extract_specs_params))
    else:
        raise ValueError('Unsupported specs_type %s' % specs_type)

    specs = np.expand_dims(np.array([resize_with_padding(im, target_shape) for im in specs]), -1)

    predictions = model.predict(specs)
    grouped_probs = [predictions[i::len(sounds_metadata)] for i in range(len(sounds_metadata))]
    if return_grouped:
        return grouped_probs
    predicted_labels = interpret_answers(grouped_probs, all_classes)
    return predicted_labels, specs
