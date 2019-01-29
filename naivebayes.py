import csv
import math
import pandas as pd
import random

df = pd.read_csv(
    r'C:\Users\Andre\Google Drive\Master Studium Bamberg\AngewandteInformatik\2nd - WS18\KogSys-ML-M\Assignment2\nb_mushrooms_training.csv')
dataset = df.drop(columns='class')
dataset_in_vectors_training = dataset.values.tolist()

df_test = pd.read_csv(
    r'C:\Users\Andre\Google Drive\Master Studium Bamberg\AngewandteInformatik\2nd - WS18\KogSys-ML-M\Assignment2\nb_mushrooms_test.csv')
test_dataset = df_test.drop(columns='class')
dataset_in_vectors_test = test_dataset.values.tolist()


def naiveBayesTrain():
    global algorithmNegatives
    algorithmNegatives = 0
    global algorithmPositives
    algorithmPositives = 0

    # Declaration of possible values for each column
    cap_shapes = {'b', 'c', 'x', 'f', 'k', 's'}
    dict_cap_shape_eatable = dict()
    dict_cap_shape_poisonous = dict()

    cap_surfaces = {'f', 'g', 'y', 's'}
    dict_cap_surface_eatable = dict()
    dict_cap_surface_poisonous = dict()

    bruises = {'t', 'f'}
    dict_bruises_eatable = dict()
    dict_bruises_poisonous = dict()

    gill_sizes = {'b', 'n'}
    dict_gill_size_eatable = dict()
    dict_gill_size_posiouns = dict()

    gill_spacing = {'c', 'w', 'n'}
    dict_gill_spacing_eatable = dict()
    dict_gill_spacing_poisonous = dict()

    # Calculating accuracy for each value of cap_shape
    for cap_shape in cap_shapes:
        nr_total_cap_shape = len(df[['class', 'cap-shape']][df['cap-shape'] == cap_shape])
        nr_eatable_cap_shape = len(df[['class', 'cap-shape']][df['cap-shape'] == cap_shape][df['class'] == 'e'])
        nr_poisonous_cap_shape = len(df[['class', 'cap-shape']][df['cap-shape'] == cap_shape][df['class'] == 'p'])
        dict_cap_shape_eatable[cap_shape] = nr_eatable_cap_shape / nr_total_cap_shape
        dict_cap_shape_poisonous[cap_shape] = nr_poisonous_cap_shape / nr_total_cap_shape

    # Calculating accuracy for each value of cap_surface
    for cap_surface in cap_surfaces:
        nr_total_cap_surface = len(df[['class', 'cap-surface']][df['cap-surface'] == cap_surface])
        nr_eatable_cap_surface = len(df[['class', 'cap-surface']][df['cap-surface'] == cap_surface][df['class'] == 'e'])
        nr_poisonous_cap_surface = len(
            df[['class', 'cap-surface']][df['cap-surface'] == cap_surface][df['class'] == 'p'])
        dict_cap_surface_eatable[cap_surface] = nr_eatable_cap_surface / nr_total_cap_surface
        dict_cap_surface_poisonous[cap_surface] = nr_poisonous_cap_surface / nr_total_cap_surface

    # Calculating accuracy for each value of bruises
    for bruise in bruises:
        nr_total_bruises = len(df[['class', 'bruises']][df['bruises'] == bruise])
        nr_eatable_bruises = len(df[['class', 'bruises']][df['bruises'] == bruise][df['class'] == 'e'])
        nr_poisonous_bruises = len(df[['class', 'bruises']][df['bruises'] == bruise][df['class'] == 'p'])
        dict_bruises_eatable[bruise] = nr_eatable_bruises / nr_total_bruises
        dict_bruises_poisonous[bruise] = nr_poisonous_bruises / nr_total_bruises

    # Calculating accuracy for each value of gill_spacing
    for gill_space in gill_spacing:
        nr_total_gill_spacing = len(df[['class', 'gill-spacing']][df['gill-spacing'] == gill_space])
        nr_eatable_gill_spacing = len(
            df[['class', 'gill-spacing']][df['gill-spacing'] == gill_space][df['class'] == 'e'])
        nr_poisonous_gill_spacing = len(
            df[['class', 'gill-spacing']][df['gill-spacing'] == gill_space][df['class'] == 'p'])
        if nr_eatable_gill_spacing == 0 & nr_poisonous_gill_spacing == 0:
            dict_gill_spacing_eatable[gill_space] = 0.5
            dict_gill_spacing_poisonous[gill_space] = 0.5
        else:
            dict_gill_spacing_eatable[gill_space] = nr_poisonous_gill_spacing / nr_total_gill_spacing
            dict_gill_spacing_poisonous[gill_space] = nr_eatable_gill_spacing / nr_total_gill_spacing

    # Calculating accuracy for each value of gill_sizes
    for gill_size in gill_sizes:
        nr_total_gill_size = len(df[['class', 'gill-size']][df['gill-size'] == gill_size])
        nr_eatable_gill_size = len(df[['class', 'gill-size']][df['gill-size'] == gill_size][df['class'] == 'e'])
        nr_poisonous_gill_size = len(df[['class', 'gill-size']][df['gill-size'] == gill_size][df['class'] == 'p'])
        dict_gill_size_eatable[gill_size] = nr_eatable_gill_size / nr_total_gill_size
        dict_gill_size_posiouns[gill_size] = nr_poisonous_gill_size / nr_total_gill_size

    # Number of eatable mushrooms
    nr_eatable = df_test['class'][df_test['class'] == 'e'].count()

    # Number of poisonous mushrooms
    nr_poisonous = df_test['class'][df_test['class'] == 'p'].count()

    # Total
    total = df_test['class'].count()

    # Probability that a mushroom is eatable
    prob_eatable = nr_eatable / total

    # Probability that a mushroom is poisonous
    prob_poisonous = nr_poisonous / total

    for i in range(len(dataset_in_vectors_test)):
        # Vector
        vector = dataset_in_vectors_test[i]

        nb_likelihood_eatable = (prob_eatable * (
                dict_cap_shape_eatable[vector[0]] * dict_cap_surface_eatable[vector[1]] * dict_bruises_eatable[
            vector[2]] * dict_gill_spacing_eatable[vector[3]] * dict_gill_size_eatable[vector[4]]))
        nb_likelihood_poisonous = (prob_poisonous * (
                dict_cap_shape_poisonous[vector[0]] * dict_cap_surface_poisonous[vector[1]] * dict_bruises_poisonous[
            vector[2]] * dict_gill_spacing_poisonous[vector[3]] * dict_gill_size_posiouns[vector[4]]))

        # Normalization of nb_prob_eatable
        normalized_nb_prob_eatable = nb_likelihood_eatable / nb_likelihood_eatable + nb_likelihood_poisonous
        # print(normalized_nb_prob_eatable)

        # Normalization of nb_prob_poisonous
        normalized_nb_prob_poisonous = nb_likelihood_poisonous / nb_likelihood_eatable + nb_likelihood_poisonous
        # print(normalized_nb_prob_poisonous)

        if normalized_nb_prob_eatable > normalized_nb_prob_poisonous:
            algorithmNegatives += 1
        else:
            # The relevant class (the 'positive' class) shall be poisonous(p)
            algorithmPositives += 1

    # Variables for calculating the accuracy, the precision and the recall
    truePositives = (nr_poisonous + algorithmPositives) - nr_poisonous
    trueNegatives = (nr_eatable + algorithmNegatives) - nr_eatable
    falsePositives = algorithmPositives-nr_poisonous if algorithmPositives-nr_poisonous >= 0 else 0
    falseNegatives = algorithmNegatives-nr_eatable if algorithmNegatives-nr_eatable >= 0 else 0

    # Accuracy
    accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
    print("Accuracy: %3.8f" % accuracy)
    # Precision
    precision = truePositives / (truePositives + falsePositives)
    print("Precision: %.8f" % precision)
    # Recall
    recall = truePositives / (truePositives + falseNegatives)
    print("Recall: %.8f" % recall)



def main():
    print(naiveBayesTrain())


if __name__ == "__main__":
    main()

