from .spectral import PopulationInitializer
import numpy as np
import scipy
import scipy.special


class MomentInitializer(PopulationInitializer):
    @staticmethod
    def BinSearch(fun,
                  y):  # find x s.t fun(x)=y, function must be strictly increasing with range (0,infinity), fun(0) = 0
        if y < 0.0013846494270530422:
            return 0.001
        if y > 0.49916666877531185:
            return 100.
        # if y == 0:
        #     return 0
        x_hi = 1
        x_low = 0
        while 1:
            f = fun(x_hi)
            if f < y:
                x_low = x_hi
                x_hi *= 2
            elif f == y:
                return x_hi
            else:
                break
        for i in range(10):
            x_mid = (x_low + x_hi) / 2
            f_mid = fun(x_mid)
            if y < f_mid:
                x_hi = x_mid
            elif y == f_mid:
                return x_mid
            else:
                x_low = x_mid
        return (x_low + x_hi) / 2

    @staticmethod
    def BetaErrorRate(x):
        if x == 0:
            return 0
        return -(1 / 6) * x * (x * np.pi * np.pi - 12 * np.log(2) + 12 * x * scipy.special.spence(1 + np.exp(-1 / x)))

    @staticmethod
    def BetaSearch(y):
        if y <= 0.5:
            return MomentInitializer.BinSearch(MomentInitializer.BetaErrorRate, y)
        else:
            return -MomentInitializer.BinSearch(MomentInitializer.BetaErrorRate, 1 - y)

    def get_initialization_point(self):
        s_est_btl = super(MomentInitializer, self).get_initialization_point()

        betaEstimates = np.zeros(len(self.data_pack.beta_true))
        s_max = max(s_est_btl)
        s_min = min(s_est_btl)
        for k in range(len(self.data_pack.count_mat)):
            data = self.data_pack.count_mat[k]
            incorrect = 0
            total = 0
            for i in range(len(s_est_btl)):
                for j in range(len(s_est_btl)):
                    d = data[i][j]
                    total += d
                    if s_est_btl[i] > s_est_btl[j]:
                        incorrect += d
            ratio = incorrect / total
            betaEstimates[k] = MomentInitializer.BetaSearch(ratio) * (s_max - s_min)

        return s_est_btl, betaEstimates


class MLInitializer(PopulationInitializer):
    @staticmethod
    def computeLogLikelihood(s, gamma, comps):
        LL = 0
        for comp in comps:
            # s[i] is winner
            si = s[comp[0]]
            sj = s[comp[1]]
            prefered = comp[2]
            LL += - np.log(1 + np.exp(-(prefered * (si - sj) + (1 - prefered) * (sj - si)) * gamma))
        return LL

    @staticmethod
    def computeGrad(s, gamma, comps):
        LL = 0
        for comp in comps:
            si = s[comp[0]]
            sj = s[comp[1]]
            preferred = comp[2]
            deltaS = preferred * (si - sj) + (1 - preferred) * (sj - si)
            LL += deltaS / (1 + np.exp(deltaS * gamma))
        return LL

    @staticmethod
    def compute2ndD(s, gamma, comps):
        LL = 0
        for comp in comps:
            si = s[comp[0]]
            sj = s[comp[1]]
            preferred = comp[2]
            deltaS = preferred * (si - sj) + (1 - preferred) * (sj - si)
            LL -= 1 / 4 * deltaS ** 2 / np.cosh(deltaS * gamma / 2)
        return LL

    @staticmethod
    def BetaMLEstimate(s, data, init=1):
        comps = []
        for i in range(np.size(data, 0)):
            for j in range(np.size(data, 1)):
                if data[i][j]:
                    # j is winner
                    comp = [j, i, 1]
                    comps = comps + [comp * int(data[i][j])]
        gamma_old = 1 / init
        # print('--------------------')
        while 1:
            gamma = gamma_old - MLInitializer.computeGrad(s, gamma_old, comps) / MLInitializer.compute2ndD(s, gamma_old, comps)
            # print(gamma)
            if (gamma - gamma_old) / gamma_old < 0.001:
                break
            gamma_old = gamma
        return 1 / gamma

    def get_initialization_point(self):
        s_est_btl = super(MLInitializer, self).get_initialization_point()

        betaEstimatesML = np.zeros(len(self.data_pack.beta_true))
        for k in range(len(self.data_pack.count_mat)):
            data = self.data_pack.count_mat[k]
            betaEstimatesML[k] = MLInitializer.BetaMLEstimate(s_est_btl, data)

        return s_est_btl, betaEstimatesML
