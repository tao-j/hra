import csv
import numpy as np


class ReadingLevelDataset():
    def __init__(self, path="wsdm_rankagg_2013_readability_crowdflower_data.csv"):
        self.s_true, self.eta_true, self.count_mat, self.data_cnt = self.load_data(path)

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
        all_judges = []
        data_cnt = {}

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
                if np.random.random() < 0.33333:
                    sign = 1
                else:
                    sign = 1
                judge_set[judge_k] = sign * judge_id
                judge_id += 1
                all_judges.append(cols[7])

            for idx, col in enumerate(cols):
                try:
                    if idx == 9:
                        if col:
                            jk = judge_set[judge_k]
                            go = 0
                            if col[8] == 'A':
                                go = 1
                            elif col[8] == 'B':
                                go = -1
                            elif col != "I don't know or can't decide.":
                                print(col, dd)
                            else:
                                idk += 1

                            if jk < 0:
                                go = -go
                                jk = -jk
                            if go == 1:
                                count_mat[jk][item_set[item_j]][item_set[item_i]] += 1
                                tp = (item_set[item_j], item_set[item_i], jk)
                                if tp not in data_cnt:
                                    data_cnt[tp] = 0
                                data_cnt[tp] += 1
                                useful += 1
                            if go == -1:
                                count_mat[jk][item_set[item_i]][item_set[item_j]] += 1
                                tp = (item_set[item_i], item_set[item_j], jk)
                                if tp not in data_cnt:
                                    data_cnt[tp] = 0
                                data_cnt[tp] += 1
                                useful += 1
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

        return np.array(all_scores), all_judges, count_mat, data_cnt


if __name__ == '__main__':
    ds = ReadingLevelDataset()
    print(ds.count_mat.sum(axis=0))
    # print(ds.s_true)
    s_true = ds.s_true.tolist()
    open('doc_info.txt', 'w').write('\n'.join(list(
        map(lambda value, idx: '{} {}'.format(idx+1, value), s_true, range(len(s_true))))))

    eta_true = ds.eta_true
    open('annotator_info.txt', 'w').write('\n'.join(list(
        map(lambda value, idx: '{} {} {}'.format(idx+1, idx+1, value), eta_true, range(len(eta_true))))))

    # print(ds.data_cnt)
    mul_list = []
    for k, v in ds.data_cnt.items():
        if v > 1:
            print(k, v)
            mul_list.append(k)
    open('all_pair.txt', 'w').write(
        '\n'.join(
            list(
                map(
                    lambda lose, win, anno_id: '{} {} {}'.format(
                        anno_id+1, win+1, lose+1
                    ), *list(
                        zip(*ds.data_cnt.keys(), *mul_list)
                    )
                )
            )
        )
    )

