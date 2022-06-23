from src.audiodata_processing.process_waves import denoise_recording, butter_highpass_filter
from src.audiodata_processing.extract_features_from_wave import extract_melspecs
from collections import defaultdict
import numpy as np


def txt_to_rec_labels(rec, txt, sr):
    true_rec_labels = np.zeros(len(rec))
    true_sound_inds = []
    for s, e, cl in txt.values:
        if cl not in ['noise', 'ns']:
            s_rec, e_rec = int(sr * s), int(sr * e)
            true_rec_labels[s_rec: e_rec] = 1
            true_sound_inds.append((s_rec, e_rec))
    return true_rec_labels, true_sound_inds


def run_splitter(rec, model, all_params_dict):

    if all_params_dict['rec_denoising']['use_rec_denoising']:
        rec = denoise_recording(rec,  all_params_dict['sr'], **all_params_dict['rec_denoising']['rec_denoising_params'])

    rec = butter_highpass_filter(rec,  all_params_dict['features']['frequency_threshold'],  all_params_dict['sr'])
    rec_sounds = [rec[s: s + all_params_dict['features']['resolution']]
                  for s in range(0, len(rec) - all_params_dict['features']['resolution'] + 1,
                                 all_params_dict['features']['step'])]
    rec_melspecs = np.array(extract_melspecs(rec_sounds, all_params_dict['sr'], all_params_dict['features']['n_fft'],
                                             all_params_dict['features']['n_mel']))

    predictions_for_each_pixel = all_params_dict['features']['resolution'] / all_params_dict['features']['step']
    rec_predictions = model.predict(np.reshape(rec_melspecs, (*np.shape(rec_melspecs), 1)))
    pred_rec_probs = np.zeros(len(rec))
    for sound_pr, ind in zip(rec_predictions, range(0, len(rec) - all_params_dict['features']['resolution'] + 1,
                                                    all_params_dict['features']['step'])):
        pred_rec_probs[ind: ind + all_params_dict['features']['resolution']] += sound_pr[0] / predictions_for_each_pixel
    return rec_predictions, pred_rec_probs


def map_prediction_to_sounds(pred_rec_probs, th, min_length=512):
    pred_rec_labels = np.zeros(len(pred_rec_probs))
    pred_sound_inds = []
    prev_ind = -1
    current_window = None
    in_sound = False
    for ind, lbl in enumerate(pred_rec_probs):
        if lbl >= th:
            if in_sound:
                current_window = (current_window[0], ind)
            else:
                in_sound = True
                current_window = (ind, ind + 1)
        else:
            if in_sound:
                in_sound = False
                if current_window[1] - current_window[0] >= min_length:
                    pred_sound_inds.append(current_window)
                    pred_rec_labels[current_window[0]: current_window[1]] = 1

            else:
                continue
    return pred_rec_labels, pred_sound_inds


def study_precision(pred_sound_inds, true_pixel_to_sound_index, true_sound_ind_to_pixels,
                    full_cover_th=0.85, partial_cover_th=0.5):
    results = []
    results_count = defaultdict(lambda: 0)

    for s_int, e_int in pred_sound_inds:

        sounds_inside = defaultdict(lambda: 0)

        for i in range(s_int, e_int):
            ind = true_pixel_to_sound_index[i]
            if ind > 0:
                sounds_inside[ind] += 1

        sounds_fully_covered = 0
        sounds_partly_covered = 0

        for ind, val in sounds_inside.items():
            length = true_sound_ind_to_pixels[ind][1] - true_sound_ind_to_pixels[ind][0]
            coverage = val / length
            if coverage >= full_cover_th:
                sounds_fully_covered += 1
            elif partial_cover_th <= coverage:
                sounds_partly_covered += 1

        str_result = '%dfull_%dpartial' % (sounds_fully_covered, sounds_partly_covered)
        results.append(str_result)
        results_count[str_result] += 1
    return results, results_count


def study_recall(true_sound_inds, pred_rec_labels, pred_pixel_to_sound_index,
                 detected_th=0.8, partly_detected_th=0.5, partial_cover_th=.3):
    results = []
    results_count = defaultdict(lambda: 0)

    per_sound_coverages = []

    for s_int, e_int in true_sound_inds:

        length = e_int - s_int
        coverage = pred_rec_labels[s_int: e_int].mean()
        if coverage < partly_detected_th:
            str_result = 'missed'
        else:
            if coverage >= detected_th:
                str_base = 'fully'
            else:
                str_base = 'partly'

            sounds_inside = defaultdict(lambda: 0)
            for i in range(s_int, e_int):
                ind = pred_pixel_to_sound_index[i]
                if ind > 0:
                    sounds_inside[ind] += 1

            overlapping_sounds = 0

            for ind, val in sounds_inside.items():
                ratio = val / length
                if ratio >= partial_cover_th:
                    overlapping_sounds += 1

            str_result = '%s_in_%d' % (str_base, overlapping_sounds)

        results.append(str_result)
        results_count[str_result] += 1
    return results, results_count

