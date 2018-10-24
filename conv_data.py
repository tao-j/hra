import csv
import numpy as np


class ReadingLevelDataset():
    def __init__(self, path="wsdm_rankagg_2013_readability_crowdflower_data.csv"):
        self.s_true, self.count_mat = self.load_data(path)

    def load_data(self, path):
        f = open(path)
        f.readline()
        lines = csv.reader(f)

        count_mat = np.zeros((624, 490, 490))

        useful = 0
        same = 0
        idk = 0
        empty = 0

        item_set = dict()
        judge_set = dict()
        doc_id = 0
        judge_id = 0
        all_scores = []
        all_accs = []

        for dd, cols in enumerate(lines):
            if len(cols) != 15:
                print('bad row', cols, dd)
                continue

            if cols[10] == cols[11]:
                same += 1

            item_j = cols[13]
            item_i = cols[12]
            judge_k = cols[8]
            if item_j not in item_set:
                item_set[item_j] = doc_id
                all_scores.append(int(cols[11]))
                doc_id += 1
            else:
                assert all_scores[item_set[item_j]] == int(cols[11])
            if item_i not in item_set:
                item_set[item_i] = doc_id
                all_scores.append(int(cols[10]))
                doc_id += 1
            else:
                assert all_scores[item_set[item_i]] == int(cols[10])
            if judge_k not in judge_set:
                judge_set[judge_k] = judge_id
                judge_id += 1

            for idx, col in enumerate(cols):
                try:
                    if idx == 9:
                        if col:
                            if col[8] == 'A':
                                count_mat[judge_set[judge_k]][item_set[item_j]][item_set[item_i]] += 1
                                useful += 1
                            elif col[8] == 'B':
                                count_mat[judge_set[judge_k]][item_set[item_i]][item_set[item_j]] += 1
                                useful += 1
                            elif col != "I don't know or can't decide.":
                                print(col, dd)
                            else:
                                idk += 1
                        else:
                            idk += 1
                            empty += 1

                except IndexError:
                    print(col)
                    exit()

            # 8 judge_name
            # 9 str[8] A(i>j) B(j>i) k(unknown)
            # 12 item_name1
            # 13 item_name2
            # 10 item1 score
            # 11 item2 score

        print('useful', useful, 'including two items with same score', same)
        print('idk', idk, 'empty', empty)
        print('n_doc', len(item_set))
        print('n_judge', len(judge_set))

        return np.array(all_scores), count_mat


if __name__ == '__main__':
    print(ReadingLevelDataset().count_mat.sum(axis=0))
    print(ReadingLevelDataset().s_true)