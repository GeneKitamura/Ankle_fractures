import numpy as np
from tf_utils.tf_metrics import model_metrics

# for N x view_num x height x weight x channel
def split_into_single_list(inp_x, inp_y, view_num):
    single_list = []
    list_labels = []
    inp_x = inp_x.astype(np.float32)
    inp_y = inp_y.astype(np.int64)
    pt = [np.split(x, view_num, axis=0) for x in inp_x]
    # now [N x [3 x (1 x 300 x 300 x 1) ] ]

    for three_views, label in zip(pt, inp_y):
        for one_view in three_views:
            one_view = np.squeeze(one_view, axis=0)
            single_list.append(one_view)
            list_labels.append(label)

    return single_list, list_labels

def vote_3_views(ordered_single_preds, ordered_labels, views=3):
    num_cases = int(len(ordered_single_preds) / views)
    voted_case = []
    true_label = []

    for i in range(num_cases):
        tmp = np.round(np.sum(ordered_single_preds[i * views: (i + 1) * views]) / views).astype(np.int64)
        voted_case.append(tmp)
        true_label.append(ordered_labels[i*views])

    return np.array(voted_case, np.float32), np.array(true_label, np.int64)

def output_metrics_from_preds(preds, labels):
    sensitivity, specificity, PPV, NPV, accuracy, _false_positives_idx, _false_negatives_idx, total_num = model_metrics(preds, labels)

    print('For the images: sensitivity/specificity are {0:.3g} / {1:.3g}.  The PPV/NPV are {2:.3g} / {3:.3g}.'.format(sensitivity, specificity, PPV, NPV))
    print('accuracy is {0:.3g} for case # of {1}'.format(accuracy, total_num))

def model_batch_testing(curr_path, inf_model, cases, labels):
    total_models = {}

    for i, model in enumerate(curr_path):
        curr_mod = 'model_{}'.format(i)
        print(curr_mod)
        the_model = inf_model()
        the_model.create_sess_graph(model)
        the_model.test_model(cases, labels)
        the_model.votes, the_model.voted_labels = vote_3_views(the_model.predictions, labels)
        output_metrics_from_preds(the_model.votes, the_model.voted_labels)
        total_models[curr_mod] = the_model
        print('\n')

    return total_models

def vote_and_output(preds, labels, n_views=3):
    a_pred, a_label = vote_3_views(preds, labels, n_views)
    output_metrics_from_preds(a_pred, a_label)

# Needs to be odd number of models so ties don't happen.
def ensemble_votes(models_list):
    tot_sum = np.zeros_like(models_list[0].predictions.shape[0])
    N = len(models_list)
    for i in models_list:
        tot_sum = np.add(i.predictions, tot_sum)
    combined_votes = np.round(tot_sum / N)
    return combined_votes