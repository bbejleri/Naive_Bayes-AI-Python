import csv
import math
import pandas as pd
import random

df = pd.read_csv(r'C:\Users\German\Desktop\Machine Learning\Sprint 2\First Assignment\nb_mushrooms_training.csv')
dataset = df.drop(columns='class')
dataset_in_vectors_training = dataset.values.tolist()

df_2 = pd.read_csv(r'C:\Users\German\Desktop\Machine Learning\Sprint 2\First Assignment\nb_mushrooms_test.csv')
test_dataset = df_2.drop(columns='class')
dataset_in_vectors_test = test_dataset.values.tolist()


def naiveBayesTrain(dataset_in_vectors_training):

    global negatives
    negatives = 0
    global positives
    positives = 0

    for i in range(len(dataset_in_vectors_training)):
        
        #Vector
        random_vector = dataset_in_vectors_training[i]

        #Number of eatable mushrooms
        nr_eatable = df['class'][df['class'] == 'e'].count()

        #Number of poisonous mushrooms
        nr_poisonous = df['class'][df['class'] == 'p'].count()

        #Total
        total = df['class'].count()

        #Probability that a mushroom is eatable
        prob_eatable = nr_eatable/total

        #Probability that a mushroom is poisonous
        prob_poisonous = nr_poisonous/total


        val_1 = random_vector[0]
        prob_cap_shape_eatable = df['cap-shape'][df['cap-shape'] == val_1].count() / nr_eatable
        prob_cap_shape_poisonous = df['cap-shape'][df['cap-shape'] == val_1].count() / nr_poisonous

        val_2 = random_vector[1]
        prob_cap_surface_eatable = df['cap-surface'][df['cap-surface'] == val_2].count() / nr_eatable
        prob_cap_surface_poisonous = df['cap-surface'][df['cap-surface'] == val_2].count() / nr_poisonous

        val_3 = random_vector[2]
        prob_bruises_eatable = df['bruises'][df['bruises'] == val_3].count() / nr_eatable
        prob_bruises_poisonous = df['bruises'][df['bruises'] == val_3].count() / nr_poisonous

        val_4 = random_vector[3]
        prob_gill_spacing_eatable = df['gill-spacing'][df['gill-spacing'] == val_4].count() / nr_eatable
        prob_gill_spacing_poisonous = df['gill-spacing'][df['gill-spacing'] == val_4].count() / nr_poisonous

        val_5 = random_vector[4]
        prob_gill_size_eatable = df['gill-size'][df['gill-size'] == val_5].count() / nr_eatable
        prob_gill_size_poisonous = df['gill-size'][df['gill-size'] == val_5].count() / nr_poisonous

        nb_likelihood_eatable = (prob_eatable *(prob_cap_shape_eatable * prob_cap_surface_eatable * prob_bruises_eatable * prob_gill_spacing_eatable * prob_gill_size_eatable))
        nb_likelihood_poisonous = (prob_poisonous *(prob_cap_shape_poisonous * prob_cap_surface_poisonous * prob_bruises_poisonous * prob_gill_spacing_poisonous * prob_gill_size_poisonous))
        #print(nb_likelihood_eatable)
        #print(nb_likelihood_poisonous)

        #Normalization of nb_prob_eatable
        normalized_nb_prob_eatable = nb_likelihood_eatable / nb_likelihood_eatable + nb_likelihood_poisonous
        #print(normalized_nb_prob_eatable)

        #Normalization of nb_prob_poisonous
        normalized_nb_prob_poisonous = nb_likelihood_poisonous / nb_likelihood_eatable + nb_likelihood_poisonous
        #print(normalized_nb_prob_poisonous)

        if(normalized_nb_prob_eatable > normalized_nb_prob_poisonous):
            #print("Classified as: Eatable")
            negatives += 1
        else:
            #print("Classified as: Poisonous")
            positives += 1
        
    print(negatives)
    print(positives)
        


def main():
 print(naiveBayesTrain(dataset_in_vectors_test))
    


if __name__ == "__main__":
    main()
