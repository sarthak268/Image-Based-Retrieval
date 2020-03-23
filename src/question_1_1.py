import numpy as np
import glob
import cv2
import scipy.ndimage as nd
import math

def convert_name2num(names, all_data):

    ind = []
    for i in names:
        ind.append(all_data.index(i))
    
    return np.asarray(ind)

def combine_arr(arr):

    arr1 = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr1.append(arr[i][j])

    return arr1

def avg(arr):

    a = 0
    b = 0
    c = 0
    for i in range(len(arr)):
        a += arr[i][0]
        b += arr[i][1]
        c += arr[i][2]
    return a, b, c

def only_names(s):

    s2 = []
    for s1 in s:
        s2.append(s1.split('/')[-1].split('.')[0])

    return s2

def feature_distance_matching(prob_img, prob_q):

    r = 0

    for i in range(len(ds)):
        for j in range(m):
            r += abs(prob_img[j, i] - prob_q[j, i]) / (1 + prob_img[j, i] + prob_q[j, i])

    return r / m

def find_quantised_value(intensity_tuple):

    r, g, b = intensity_tuple[0], intensity_tuple[1], intensity_tuple[2]
    value = int(r / 10.) + int(g / 10.) * total_bins_each + int(b / 10.) * total_bins_each * total_bins_each

    return value
		
def quantise(img):

    quantised_img = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            intensity = img[i, j, :]
            v = find_quantised_value(intensity)

            quantised_img[i, j] = v

    return quantised_img

def get_prob_matrix_element(img_ci, d, i, j):

    w, h = img_ci.shape[0], img_ci.shape[1]

    left = max(i-d, 0)
    right = min(i+d, w-1)
    bottom = max(j-d, 0)
    up = min(j+d, h-1)

    num = 0
    for i in range(left, right+1):
        num += img_ci[i, up]
        num += img_ci[i, bottom]
    for i in range(bottom, up+1):
        num += img_ci[left, i]
        num += img_ci[right, i]

    den = 2*(abs(right - left) + abs(up - bottom))

    return num, den

def get_prob_matrix(q_img):

    img_ci_dict = {}

    for i in range(m):
        img_ci_dict[i] = np.where(q_img == i, 1, 0)

    prob_mat_n = np.zeros((m, len(ds)))
    prob_mat_d = np.zeros((m, len(ds)))

    for d_i in range(len(ds)):
        d = ds[d_i]

        for i in range(q_img.shape[0]):
            for j in range(q_img.shape[1]):

                q_value = int(q_img[i, j])
                corresponding_img = img_ci_dict[q_value]
                num, den = get_prob_matrix_element(corresponding_img, d, i, j)

                prob_mat_n[q_value, d_i] += num
                prob_mat_d[q_value, d_i] += den

    final_prob_matrix = np.divide(prob_mat_n, prob_mat_d)
    final_prob_matrix = np.nan_to_num(final_prob_matrix)
    return final_prob_matrix

def get_ground_truths(file):

    gt = []

    for types in ground_truth_types:
        gt1 = []
        with open('./train/ground_truth/' + file + '_' + types + '.txt') as g:
            for line in g:
                gt1.append(line.strip('\n'))
        gt.append(gt1)
    return gt

def make_query(otherfiles, original):

    orig_mat = np.load('./database_cc/' + original)

    count = 0

    scores = []
    for i in (otherfiles):

        count += 1

        query_mat = np.load(i)
        s = feature_distance_matching(orig_mat, query_mat)
        scores.append(s)

    return scores

def main_save_database():

    img_database = (glob.glob('./images/*.jpg'))
    counter = 0
    total_images = len(img_database)

    for im in img_database:

        print ('Images Done : ' + str(counter) + ' / ' + str(total_images))

        img = cv2.imread(im)
        img = cv2.resize(img, (64, 64))
    
        quan_img = quantise(img)
        p_mat = get_prob_matrix(quan_img) 

        np.save('./database_cc/' + im.split('.')[1].split('/')[2]  + '.npy', p_mat)
        counter += 1

def main_query():

    query_database = glob.glob('./train/query/*.txt')
    all_database = glob.glob('./database_cc/*.npy')

    #all_database = all_database[:100]
    all_database_only_names = only_names(all_database)

    precisions = []
    recalls = []
    query_count = 0

    good_percentage = []
    ok_percentage = []
    junk_percentage = []

    f1 = []

    for qu in query_database:

        filename = qu.split('/')[-1].split('.')[0][:-6]

        ground_truths = get_ground_truths(filename)
        ground_truths_all = combine_arr(ground_truths)
       
        with open(qu) as q:
            for line in q:
                a = line.split(' ')[0][5:]

        original_filename = a + '.npy'
        scores = make_query(all_database, original_filename)

        sorted_scores_ind = np.argsort(scores)
        
        ns = [100, 200, 300, 400, 500]
        p = []
        r = []

        for n in ns: 
            top_n = np.array(sorted_scores_ind[:n])

            top_matches = np.array(all_database)[top_n]
            top_matches_final = []
            for i in top_matches:
                top_matches_final.append(i.split('/')[-1].split('.')[0])

            indices_predicted = convert_name2num(top_matches_final, all_database_only_names)
            
            indices_actual_good = convert_name2num(ground_truths[0], all_database_only_names)
            indices_actual_ok = convert_name2num(ground_truths[2], all_database_only_names)
            indices_actual_junk = convert_name2num(ground_truths[1], all_database_only_names)
            indices_actual_all = convert_name2num(ground_truths_all, all_database_only_names)
            
            good_percentage.append(indices_actual_good.shape[0] / indices_actual_all.shape[0])
            ok_percentage.append(indices_actual_ok.shape[0] / indices_actual_all.shape[0])
            junk_percentage.append(indices_actual_junk.shape[0] / indices_actual_all.shape[0])

            intersection_total = np.intersect1d(indices_predicted, indices_actual_all)
            
            precision = len(intersection_total) / n
            p.append(precision)
            recall = len(intersection_total) / len(ground_truths_all)       
            r.append(recall)

            f_score = (2 * precision * recall) / (precision + recall)
            f1.append(f_score)

        precisions.append([max(p), min(p), sum(p) / len(p)])
        recalls.append([max(r), min(r), sum(r) / len(r)])
        query_count += 1
        print (str(query_count) + ' / ' + str(len(query_database)))

        # if (query_count == 2):
        #     break

    print ('\n')
    print ('Precision = ',avg(precisions))
    print ('\n')
    print ('Recalls = ',avg(recalls))
    print ('\n')
    print ('F1 Score = ', sum(f1) / len(f1))
    print ('\n')

    good_p = sum(good_percentage) / len(good_percentage)
    ok_p = sum(ok_percentage) / len(ok_percentage)
    junk_p = sum(junk_percentage) / len(junk_percentage)

    print ('Good Percentage = ' +  str(good_p * 100.))
    print ('Ok Percentage = ' + str(ok_p * 100.))
    print ('Junk Percentage = ' + str(junk_p * 100))

if (__name__ == '__main__'):

    total_bins_each = 26
    m = total_bins_each**3
    ds = [2, 3]
    ground_truth_types = ['good', 'junk', 'ok']
    
	# run once for saving centers
    #save_cluster_centers()

    ######################################
    #  FOR SAVING DATABASE
    #main_save_database()
    ######################################

    ######################################
    # FOR QUERY
    main_query()
    ######################################
    
