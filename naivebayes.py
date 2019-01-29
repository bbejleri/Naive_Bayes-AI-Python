import csv
import math
import pandas as pd
import random

df = pd.read_csv(
    r'C:\Users\Andre\Google Drive\Master Studium Bamberg\AngewandteInformatik\2nd - WS18\KogSys-ML-M\Assignment2\nb_mushrooms_training.csv')
dataset = df.drop(columns='class')
dataset_in_vectors_training = dataset.values.tolist()

df_2 = pd.read_csv(
    r'C:\Users\Andre\Google Drive\Master Studium Bamberg\AngewandteInformatik\2nd - WS18\KogSys-ML-M\Assignment2\nb_mushrooms_test.csv')
test_dataset = df_2.drop(columns='class')
dataset_in_vectors_test = test_dataset.values.tolist()


def naiveBayesTrain(df):
    global algorithmEatables
    algorithmEatables = 0
    global algorithmPoisonous
    algorithmPoisonous = 0

    cap_shapes = {'b', 'c', 'x', 'f', 'k', 's'}
    dict_cap_shape_eatable = dict()
    dict_cap_shape_poisonous = dict()

    cap_surfaces = {'f', 'g', 'y', 's'}
    dict_cap_surface_eatable = dict()
    dict_cap_surface_poisonous = dict()

    bruises = {'t', 'f'}
    dict_bruises_eatable = dict()
    dict_bruises_poisonous = dict()

    grill_sizes = {'b', 'n'}
    dict_grill_size_eatable = dict()
    dict_grill_size_posiouns = dict()

    grill_spacing = {'c', 'w', 'n'}
    dict_grill_spacing_eatable = dict()
    dict_grill_spacing_poisonous = dict()

    for cap_shape in cap_shapes:
        nr_total_cap_shape = len(df[['class', 'cap-shape']][df['cap-shape'] == cap_shape])
        nr_eatable_cap_shape = len(df[['class', 'cap-shape']][df['cap-shape'] == cap_shape][df['class'] == 'e'])
        nr_poisonous_cap_shape = len(df[['class', 'cap-shape']][df['cap-shape'] == cap_shape][df['class'] == 'p'])
        dict_cap_shape_eatable[cap_shape] = nr_eatable_cap_shape / nr_total_cap_shape
        dict_cap_shape_poisonous[cap_shape] = nr_poisonous_cap_shape / nr_total_cap_shape

    for cap_surface in cap_surfaces:
        nr_total_cap_surface = len(df[['class', 'cap-surface']][df['cap-surface'] == cap_surface])
        nr_eatable_cap_surface = len(df[['class', 'cap-surface']][df['cap-surface'] == cap_surface][df['class'] == 'e'])
        nr_poisonous_cap_surface = len(
            df[['class', 'cap-surface']][df['cap-surface'] == cap_surface][df['class'] == 'p'])
        dict_cap_surface_eatable[cap_surface] = nr_eatable_cap_surface / nr_total_cap_surface
        dict_cap_surface_poisonous[cap_surface] = nr_poisonous_cap_surface / nr_total_cap_surface

    for bruise in bruises:
        nr_total_bruises = len(df[['class', 'bruises']][df['bruises'] == bruise])
        nr_eatable_bruises = len(df[['class', 'bruises']][df['bruises'] == bruise][df['class'] == 'e'])
        nr_poisonous_bruises = len(df[['class', 'bruises']][df['bruises'] == bruise][df['class'] == 'p'])
        dict_bruises_eatable[bruise] = nr_eatable_bruises / nr_total_bruises
        dict_bruises_poisonous[bruise] = nr_poisonous_bruises / nr_total_bruises

    for gill_space in grill_spacing:
        nr_total_gill_spacing = len(df[['class', 'gill-spacing']][df['gill-spacing'] == gill_space])
        nr_eatable_gill_spacing = len(
            df[['class', 'gill-spacing']][df['gill-spacing'] == gill_space][df['class'] == 'e'])
        nr_poisonous_gill_spacing = len(
            df[['class', 'gill-spacing']][df['gill-spacing'] == gill_space][df['class'] == 'p'])
        if nr_eatable_gill_spacing == 0 & nr_poisonous_gill_spacing == 0:
            dict_grill_spacing_eatable[gill_space] = 0.5
            dict_grill_spacing_poisonous[gill_space] = 0.5
        else:
            dict_grill_spacing_eatable[gill_space] = nr_poisonous_gill_spacing / nr_total_gill_spacing
            dict_grill_spacing_poisonous[gill_space] = nr_eatable_gill_spacing / nr_total_gill_spacing

    for gill_size in grill_sizes:
        nr_total_gill_size = len(df[['class', 'gill-size']][df['gill-size'] == gill_size])
        nr_eatable_gill_size = len(df[['class', 'gill-size']][df['gill-size'] == gill_size][df['class'] == 'e'])
        nr_poisonous_gill_size = len(df[['class', 'gill-size']][df['gill-size'] == gill_size][df['class'] == 'p'])
        dict_grill_size_eatable[gill_size] = nr_eatable_gill_size / nr_total_gill_size
        dict_grill_size_posiouns[gill_size] = nr_poisonous_gill_size / nr_total_gill_size

    # Number of eatable mushrooms
    nr_eatable = df['class'][df['class'] == 'e'].count()

    # Number of poisonous mushrooms
    nr_poisonous = df['class'][df['class'] == 'p'].count()

    # Total
    total = df['class'].count()

    # Probability that a mushroom is eatable
    prob_eatable = nr_eatable / total

    # Probability that a mushroom is poisonous
    prob_poisonous = nr_poisonous / total

    for i in range(len(dataset_in_vectors_test)):
        # Vector
        vector = dataset_in_vectors_test[i]

        nb_likelihood_eatable = (prob_eatable * (
                dict_cap_shape_eatable[vector[0]] * dict_cap_surface_eatable[vector[1]] * dict_bruises_eatable[
            vector[2]] * dict_grill_spacing_eatable[vector[3]] * dict_grill_size_eatable[vector[4]]))
        nb_likelihood_poisonous = (prob_poisonous * (
                dict_cap_shape_poisonous[vector[0]] * dict_cap_surface_poisonous[vector[1]] * dict_bruises_poisonous[
            vector[2]] * dict_grill_spacing_poisonous[vector[3]] * dict_grill_size_posiouns[vector[4]]))

        # Normalization of nb_prob_eatable
        normalized_nb_prob_eatable = nb_likelihood_eatable / nb_likelihood_eatable + nb_likelihood_poisonous
        # print(normalized_nb_prob_eatable)

        # Normalization of nb_prob_poisonous
        normalized_nb_prob_poisonous = nb_likelihood_poisonous / nb_likelihood_eatable + nb_likelihood_poisonous
        # print(normalized_nb_prob_poisonous)

        if (normalized_nb_prob_eatable > normalized_nb_prob_poisonous):
            # print("Classified as: Eatable")
            algorithmEatables += 1
        else:
            # print("Classified as: Poisonous")
            algorithmPoisonous += 1


    print(algorithmEatables)
    print(algorithmPoisonous)


def main():
    print(naiveBayesTrain(df))


if __name__ == "__main__":
    main()

