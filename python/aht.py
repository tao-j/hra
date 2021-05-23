import numpy as np
import random
from python.cmpsort import CmpSort


class ActiveRank:
    def __init__(self, N, M, delta, s, gamma):
        self.N = N
        self.M = M
        self.cM = range(0, M)
        self.s = s
        self.gamma = gamma
        self.delta = delta
        self.cmp_sort = CmpSort(s, delta)

        self.rank_sample_complexity = 0

    def eliminate_user(self, eps=0.1, delta=0.1):
        pass

    def rank(self):
        while not self.cmp_sort.done:
            pair = self.cmp_sort.next_pair()
            assert (0 <= pair[0] <= self.cmp_sort.n_intree)
            assert (-1 <= pair[1] <= self.cmp_sort.n_intree)
            if pair[1] == -1:
                self.cmp_sort.feedback(1)
            elif pair[1] == self.cmp_sort.n_intree:
                self.cmp_sort.feedback(0)
            else:
                pack_a = self.atc(pair[0], pair[1], self.cmp_sort.epsilon_atc_param, self.cmp_sort.delta_atc_param,
                                  self.cmp_sort.ranked_list, self.s, self.gamma)
                pack_b = self.cmp_sort.feedback(pack_a[0])
                self.post_atc(pack_a, pack_b)

        return self.rank_sample_complexity, self.cmp_sort.ranked_list

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        pass

    def post_atc(self, pack_a, pack_b):
        pass

    def init_user_counter(self):
        pass

    def update_user_counter(self):
        pass


class TwoStageSimultaneousActiveRank(ActiveRank):
    def __init__(self, N, M, delta, s, gamma):
        super().__init__(N, M, delta, s, gamma)
        self.n_t = np.zeros(M)
        self.s_t = 0

    def post_atc(self, pack_a, pack_b):
        y, bn, r = pack_a
        self.s_t += r
        self.n_t += bn
        self.rank_sample_complexity += r * len(self.cM)
        self.cM = self.eliminate_user(delta=delta)

    def eliminate_user(self, eps=0.1, delta=0.1):
        if len(self.cM) == 1:
            return self.cM
        s_max = int(np.ceil(2 / eps / eps * np.log2(len(self.cM) / delta)))
        if self.s_t > s_max:
            mu_t = self.n_t / self.s_t
            i_best = np.argmax(mu_t)
            self.cM = [i_best]
        return self.cM

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        """
        Do AttemptToCompare in rounds. One round asks every user once.
        """
        w = 0
        m_t = len(self.cM)
        b_max = np.ceil(1. / 2 / m_t / eps ** 2 * np.log(2 / delta))
        bn = np.zeros(self.M)
        p = 0.5
        r = 0
        for t in range(1, int(b_max)):
            for u in self.cM:
                # s_i = original_s[i] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
                # s_j = ranked_s[j] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
                pij = np.exp(gamma[u] * original_s[i]) / (
                        np.exp(gamma[u] * original_s[i]) + np.exp(gamma[u] * ranked_s[j]))
                y = 1 if np.random.random() < pij else 0
                if y == 1:
                    w += 1
                    bn[u] += 1
            r = t
            b_t = np.sqrt(1. / 2 / (r + 1) / m_t * np.log(np.pi ** 2 * (r + 1) ** 2 / 3 / delta))
            p = w / r / len(self.cM)
            if p > 0.5 + b_t:
                break
            if p < 0.5 - b_t:
                break

        atc_y = 1 if p > 0.5 else 0
        bn = bn if p > 0.5 else r - bn
        return atc_y, bn, r


class UnevenUCBActiveRank(ActiveRank):
    def __init__(self, N, M, delta, s, gamma):
        super().__init__(N, M, delta, s, gamma)
        # number of times user is asked
        self.bs = np.zeros(M)
        # number of times user is correct
        self.bn = np.zeros(M)
        # temp matrix list holding user response
        self.A = self.create_mat(N, M)

    @staticmethod
    def create_mat(N, M):
        A = []
        for i in range(M):
            A.append(np.zeros((N, N)))
        return A

    def post_atc(self, pack_a, pack_b):
        inserted, inserted_place = pack_b
        if inserted:
            assert inserted_place != -1
            inserted_idx = len(self.cmp_sort.ranked_list)
            B = np.zeros((self.N, self.N))
            for i in range(inserted_idx):
                if inserted_place > i:
                    B[i, inserted_place] = 1
                elif inserted_place < i:
                    B[inserted_place, i] = 1
            for u in range(self.M):
                self.bn[u] += sum(sum(self.A[u] * B))
            self.A = self.create_mat(self.N, self.M)
            self.eliminate_user()

    def atc(self, i, j, eps, delta, ranked_s, original_s, gamma):
        m_t = len(self.cM)
        t_max = int(np.ceil(1. / 2 / (eps ** 2) * np.log(2 / delta)))
        p = 0.5
        w = 0
        for t in range(1, t_max + 1):
            u = int(np.floor(np.random.random() * m_t))
            self.bs[u] += 1
            # s_i = original_s[i] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
            # s_j = ranked_s[j] + np.random.gumbel(0.5772 * gamma[u], gamma[u])
            pij = np.exp(gamma[u] * original_s[i]) / (
                    np.exp(gamma[u] * original_s[i]) + np.exp(gamma[u] * ranked_s[j]))
            y = 1 if np.random.random() < pij else 0
            if y == 1:
                self.A[u][i, j] += 1
                w += 1
            else:
                self.A[u][j, i] += 1
            b_t = np.sqrt(1. / 2 / t * np.log(np.pi * np.pi * t * t / 3 / delta))
            p = w / t
            if p > 0.5 + b_t:
                break
            if p < 0.5 - b_t:
                break

        atc_y = 1 if p > 0.5 else 0
        return atc_y, self.A, self.bs

    def eliminate_user(self, eps=0.1, delta=0.1):
        mu = self.bn / self.bs
        # TODO: log2 ?
        r = np.sqrt(np.log2(2 * len(self.cM) / self.delta) / 2 / self.bs)
        bucb = mu + r
        blcb = mu - r
        to_remove = set()
        for u in range(len(self.cM)):
            for up in range(len(self.cM)):
                if bucb[u] < blcb[up]:
                    to_remove.add(u)
                    break
        new_cM = []
        for u in self.cM:
            if u not in to_remove:
                new_cM.append(u)
        self.cM = new_cM


def gamma_sweep(act, repeat, eps=0.1, delta=0.1):
    random.seed(123)
    np.random.seed(123)

    # data gen method may be mentioned in Ren et al.
    thetas = []
    for i in range(100):
        li = 0.9 * np.power(1.2 / 0.8, i)
        ri = 1.1 * np.power(1.2 / 0.8, i)
        thetas.append(li + (ri - li) * (np.random.random()))
        # thetas.append(np.power(1.2 / 0.8, i))

    N = 10
    M = 9
    # for gb in [0.25, 1., 2.5]:
    #     for gg in [2.5, 5, 10]:
    # for gb in [2.5]:
    #     for gg in [2.5]:
    for gb in [0.5]:
        for gg in [5.]:

            gamma = [gg] * (M // 3) + [gb] * (M // 3 * 2)
            gamma += [gb] * (M - len(gamma))
            # s = np.linspace(1 / n, 1, n)
            s = np.log(thetas[:N])
            tts = []
            s_idx = list(range(0, len(s)))
            random.shuffle(s_idx)
            s = s[s_idx]
            # np.random.shuffle(s)

            for _ in range(repeat):
                # algo = TwoStageSimultaneousActiveRank(N, M, delta, s, gamma)
                algo = UnevenUCBActiveRank(N, M, delta, s, gamma)
                rank_sample_complexity, ranked_list = algo.rank()
                tts.append(rank_sample_complexity)
                # print(len(cM))
                a_ms = list(ranked_list)
                # print(a_ms)
                a_sorted = sorted(s)
                # print(a_sorted)

                # if len(cM) == 1:
                #     if not gamma[cM[0]] == gg:
                #         print(cM)
                #
                assert (a_ms == a_sorted)
                # print(cM)
            # print('&', int(np.average(tts)), " $\\pm$ ", int(np.std(tts)), end='\t\t'),
            print(int(np.average(tts)))
            # print('\n'.join(list(map(str, tts))))
        # print('')


if __name__ == "__main__":
    repeat = 10
    delta = 0.1
    # for delta in np.arange(0.05, 1, 0.05):
    gamma_sweep(act=0, repeat=repeat, delta=delta)
